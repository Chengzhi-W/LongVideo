import gc
import logging
from contextlib import nullcontext

from model import CausalDiffusion
from utils.dataset import cycle, LatentLMDBDataset
from utils.misc import set_seed
import torch.distributed as dist
from omegaconf import OmegaConf
import torch
import wandb
import time
import os
import math
from utils.distributed import EMA_FSDP, barrier, fsdp_wrap, fsdp_state_dict, launch_distributed_job
from pipeline import (
    CausalDiffusionInferencePipeline,
    CausalInferencePipeline,
)

class Trainer:
    def __init__(self, config):
        self.config = config
        self.step = 0

        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.causal = config.causal
        self.disable_wandb = config.disable_wandb

        # use a random seed for the training
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_seed(config.seed + global_rank)

        if self.is_main_process and not self.disable_wandb:
            wandb.login(
                host=config.wandb_host,
                key=getattr(config, "wandb_key", None) or os.environ.get("WANDB_API_KEY")
            )
            wandb.init(
                config=OmegaConf.to_container(config, resolve=True),
                name=config.config_name,
                mode="online",
                entity=config.wandb_entity,
                project=config.wandb_project,
                dir=config.wandb_save_dir
            )

        self.output_path = config.logdir

        # Step 2: Initialize the model and optimizer
        self.model = CausalDiffusion(config, device=self.device)
        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy
        )

        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy
        )

        if not config.no_visualize or config.load_raw_video:
            self.model.vae = self.model.vae.to(
                device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )

        # Step 3: Initialize the dataloader
        dataset = LatentLMDBDataset(config.data_path, max_pair=int(1e8))

        self.dataset = dataset
        self.gradient_accumulation_steps = int(
            getattr(config, "gradient_accumulation_steps", 1))
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        effective_global_batch = (
            dist.get_world_size()
            * int(config.batch_size)
            * self.gradient_accumulation_steps
        )
        configured_global_batch = int(getattr(config, "total_batch_size", effective_global_batch))
        if configured_global_batch != effective_global_batch:
            raise RuntimeError(
                f"total_batch_size={configured_global_batch}, but world_size * "
                "batch_size * gradient_accumulation_steps="
                f"{effective_global_batch}. Make them equal."
            )
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=True)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=int(getattr(config, "num_workers", 8)))

        if dist.get_rank() == 0:
            print("DATASET SIZE %d" % len(dataset))
            print("DATASET LATENT SHAPE", dataset.latents_shape)
            print(
                "AR TEACHER BATCH CONTRACT "
                f"world_size={dist.get_world_size()} "
                f"micro_batch={int(config.batch_size)} "
                f"gradient_accumulation={self.gradient_accumulation_steps} "
                f"global_batch={effective_global_batch}"
            )
        expected_shape = tuple(int(v) for v in config.image_or_video_shape[1:])
        actual_shape = tuple(int(v) for v in dataset.latents_shape[1:])
        if actual_shape != expected_shape:
            raise RuntimeError(
                f"LMDB/config shape mismatch: dataset row {actual_shape} != "
                f"image_or_video_shape {expected_shape}"
            )
        self.dataloader = cycle(dataloader)

        ##############################################################################################################
        # 6. Set up EMA parameter containers
        rename_param = (
            lambda name: name.replace("_fsdp_wrapped_module.", "")
            .replace("_checkpoint_wrapped_module.", "")
            .replace("_orig_mod.", "")
        )
        self.name_to_trainable_params = {}
        for n, p in self.model.generator.named_parameters():
            if not p.requires_grad:
                continue

            renamed_n = rename_param(n)
            self.name_to_trainable_params[renamed_n] = p
        ema_weight = config.ema_weight
        self.generator_ema = None
        if (ema_weight is not None) and (ema_weight > 0.0):
            print(f"Setting up EMA with weight {ema_weight}")
            self.generator_ema = EMA_FSDP(self.model.generator, decay=ema_weight)

        ##############################################################################################################
        # 7. (If resuming) Load the model and optimizer, lr_scheduler, ema's statedicts
        if getattr(config, "generator_ckpt", False):
            print(f"Loading pretrained generator from {config.generator_ckpt}")
            state_dict = torch.load(config.generator_ckpt, map_location="cpu")
            if "generator" in state_dict:
                state_dict = state_dict["generator"]
                fixed = {}
                for k, v in state_dict.items():
                    if k.startswith("model._fsdp_wrapped_module."):
                        k = k.replace("model._fsdp_wrapped_module.", "model.", 1)
                    fixed[k] = v
                state_dict = fixed
            elif "model" in state_dict:
                state_dict = state_dict["model"]
            elif "generator_ema" in state_dict:
                gen_sd = state_dict["generator_ema"]
                fixed = {}
                for k, v in gen_sd.items():
                    if k.startswith("model._fsdp_wrapped_module."):
                        k = k.replace("model._fsdp_wrapped_module.", "model.", 1)
                    fixed[k] = v
                state_dict = fixed
            self.model.generator.load_state_dict(state_dict, strict=True)

        ##############################################################################################################

        # Let's delete EMA params for early steps to save some computes at training and inference
        if self.step < config.ema_start_step:
            self.generator_ema = None

        self.max_grad_norm = 10.0
        self.previous_time = None
        self.delta_mean = None
        self.rtf_ema_ratio = getattr(self.config, "rtf_ema_ratio", 0.9)
        self.eval_interval = getattr(self.config, "eval_interval", 0)      # 0 => disable
        self.eval_frames = getattr(self.config, "eval_num_output_frames", 21)
        self.eval_init = getattr(self.config, "eval_num_init_frames", 3)
        self.rtf_single_gpu_batch = getattr(self.config, "rtf_single_gpu_batch", 1)
        self.given_first_chunk = getattr(self.config, "given_first_chunk", True)
        if self.eval_interval:
            self.pipeline = CausalDiffusionInferencePipeline(config, device=self.device)
            self.pipeline.generator = self.model.generator
            self.pipeline.text_encoder = self.model.text_encoder

    def save(self):
        print("Start gathering distributed model states...")
        generator_state_dict = fsdp_state_dict(
            self.model.generator)

        if self.generator_ema is not None and self.config.ema_start_step < self.step:
            state_dict = {
                "generator": generator_state_dict,
                "generator_ema": self.generator_ema.full_state_dict(self.model.generator),
            }
        else:
            state_dict = {
                "generator": generator_state_dict,
            }

        if self.is_main_process:
            os.makedirs(os.path.join(self.output_path,
                        f"checkpoint_model_{self.step:06d}"), exist_ok=True)
            torch.save(state_dict, os.path.join(self.output_path,
                       f"checkpoint_model_{self.step:06d}", "model.pt"))
            print("Model saved to", os.path.join(self.output_path,
                  f"checkpoint_model_{self.step:06d}", "model.pt"))

    def train_one_step(self, micro_batches):
        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        if len(micro_batches) != self.gradient_accumulation_steps:
            raise RuntimeError(
                "micro-batch count does not match gradient accumulation: "
                f"{len(micro_batches)} != {self.gradient_accumulation_steps}"
            )
        self.generator_optimizer.zero_grad(set_to_none=True)
        accumulated_loss = torch.zeros((), device=self.device, dtype=torch.float32)

        for micro_index, batch in enumerate(micro_batches):
            text_prompts = batch["prompts"]
            if not self.config.load_raw_video:  # precomputed latent
                clean_latent = batch["clean_latent"].to(
                    device=self.device, dtype=self.dtype)
            else:  # encode raw video to latent
                frames = batch["frames"].to(
                    device=self.device, dtype=self.dtype)

                with torch.no_grad():
                    clean_latent = self.model.vae.encode_to_latent(
                        frames).to(device=self.device, dtype=self.dtype)
            image_latent = clean_latent[:, 0:1, ]

            batch_size = len(text_prompts)
            image_or_video_shape = list(self.config.image_or_video_shape)
            image_or_video_shape[0] = batch_size

            with torch.no_grad():
                conditional_dict = self.model.text_encoder(
                    text_prompts=text_prompts)
                if not getattr(self, "unconditional_dict", None):
                    unconditional_dict = self.model.text_encoder(
                        text_prompts=[self.config.negative_prompt] * batch_size)
                    unconditional_dict = {
                        key: value.detach()
                        for key, value in unconditional_dict.items()
                    }
                    self.unconditional_dict = unconditional_dict
                else:
                    unconditional_dict = self.unconditional_dict

            should_sync = micro_index + 1 == self.gradient_accumulation_steps
            sync_context = (
                nullcontext() if should_sync else self.model.generator.no_sync()
            )
            with sync_context:
                generator_loss, _ = self.model.generator_loss(
                    image_or_video_shape=image_or_video_shape,
                    conditional_dict=conditional_dict,
                    unconditional_dict=unconditional_dict,
                    clean_latent=clean_latent,
                    initial_latent=image_latent,
                )
                if not bool(torch.isfinite(generator_loss)):
                    raise FloatingPointError(
                        f"non-finite generator loss at micro-batch {micro_index}"
                    )
                (generator_loss / self.gradient_accumulation_steps).backward()
            accumulated_loss += generator_loss.detach().float()

        generator_loss = accumulated_loss / self.gradient_accumulation_steps
        generator_grad_norm = self.model.generator.clip_grad_norm_(
            self.max_grad_norm)
        if not bool(torch.isfinite(generator_grad_norm)):
            raise FloatingPointError("non-finite generator gradient norm")
        self.generator_optimizer.step()

        self.step += 1
        generator_loss_value = generator_loss.item()
        generator_grad_norm_value = generator_grad_norm.item()
        wandb_loss_dict = {
            "generator_loss": generator_loss_value,
            "generator_grad_norm": generator_grad_norm_value,
        }

        if self.is_main_process:
            if self.step == 1 or self.step % int(self.config.log_iters) == 0:
                print(
                    "AR_TEACHER_TRAIN_STEP:"
                    f"step={self.step}:loss={generator_loss_value:.8f}:"
                    f"grad_norm={generator_grad_norm_value:.8f}:"
                    f"micro_batches={self.gradient_accumulation_steps}"
                )
            if not self.disable_wandb:
                wandb.log(wandb_loss_dict, step=self.step)

        if self.step % self.config.gc_interval == 0:
            if dist.get_rank() == 0:
                logging.info("DistGarbageCollector: Running GC.")
            gc.collect()

    def train(self):
        configured_save_steps = getattr(self.config, "save_steps", None)
        save_steps = (
            {int(step) for step in configured_save_steps}
            if configured_save_steps else None
        )
        max_steps = int(getattr(self.config, "max_steps", 0) or 0)
        last_saved_step = -1

        while max_steps <= 0 or self.step < max_steps:
            micro_batches = [
                next(self.dataloader)
                for _ in range(self.gradient_accumulation_steps)
            ]
            self.train_one_step(micro_batches)

            should_save = (
                self.step in save_steps
                if save_steps is not None
                else self.step % self.config.log_iters == 0
            )
            if (not self.config.no_save) and should_save:
                torch.cuda.empty_cache()
                self.save()
                last_saved_step = self.step
                torch.cuda.empty_cache()

            barrier()
            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    if not self.disable_wandb:
                        wandb.log({"per iteration time": current_time - self.previous_time}, step=self.step)
                    self.previous_time = current_time

        if not self.config.no_save and self.step != last_saved_step:
            torch.cuda.empty_cache()
            self.save()
            torch.cuda.empty_cache()
        if self.is_main_process:
            print(f"Reached max_steps={max_steps}; AR teacher fine-tuning complete.")
