from utils.dataset import cycle
from utils.dataset import TextDataset
from utils.distributed import barrier, fsdp_wrap, fsdp_state_dict, launch_distributed_job
from utils.misc import set_seed
import torch.distributed as dist
from omegaconf import OmegaConf
from model import OPD
import torch
import wandb
import time
import os


class Trainer:
    """
    On-Policy Distillation trainer. Trains the causal AR generator so that, on its
    OWN on-policy rollout states, it matches a frozen bidirectional Wan teacher
    (see model/opd.py OPD.opd_loss). Single-optimizer, no critic.
    """

    def __init__(self, config):
        self.config = config
        self.step = 0

        # Step 1: distributed / seed / dtype / logging.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.causal = config.causal
        self.disable_wandb = config.disable_wandb

        # seed == 0 means "random seed"; otherwise reproducible.
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()
        set_seed(config.seed + global_rank)

        if self.is_main_process and not self.disable_wandb:
            wandb.login(host=config.wandb_host, key=config.wandb_key)
            wandb.init(
                config=OmegaConf.to_container(config, resolve=True),
                name=config.logdir,
                project=config.wandb_project,
                dir=config.wandb_save_dir
            )

        self.output_path = config.logdir

        # Step 2: Initialize the model.
        assert config.trainer == "opd", "OPDTrainer only supports trainer == 'opd'"
        self.model = OPD(config, device=self.device)

        # FSDP wrap: trainable causal generator + frozen bidirectional Wan teacher
        # (real_score) + text_encoder. No critic / fake_score / EMA.
        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy,
            cpu_offload=False
        )

        self.model.real_score = fsdp_wrap(
            self.model.real_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=getattr(config, "real_score_fsdp_wrap_strategy", "size"),
            cpu_offload=False
        )

        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
            cpu_offload=getattr(config, "text_encoder_cpu_offload", False)
        )

        # VAE is needed: the rollout re-encodes the boundary frame of the last-21
        # window (and the diffusion rollout pipeline uses it only if decoding, which
        # we skip via return_video=False -- but keep it on device for safety).
        self.model.vae = self.model.vae.to(
            device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )

        # Step 3: dataloader (text prompts only).
        dataset = TextDataset(config.data_path)
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=True)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=8)
        if dist.get_rank() == 0:
            print("DATASET SIZE %d" % len(dataset))
        self.dataloader = cycle(dataloader)

        # Step 4: load the pretrained generator (the AR teacher ar_diffusion.pt).
        if getattr(config, "generator_ckpt", False):
            print(f"Loading pretrained generator from {config.generator_ckpt}")
            state_dict = torch.load(config.generator_ckpt, map_location="cpu")
            if "generator" in state_dict:
                state_dict = state_dict["generator"]
            elif "generator_ema" in state_dict:
                state_dict = state_dict["generator_ema"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]
            fixed = {}
            for k, v in state_dict.items():
                if k.startswith("model._fsdp_wrapped_module."):
                    k = k.replace("model._fsdp_wrapped_module.", "model.", 1)
                fixed[k] = v
            self.model.generator.load_state_dict(fixed, strict=True)

        self.max_grad_norm = getattr(config, "max_grad_norm_generator", 10.0)
        self.previous_time = None
        self.unconditional_dict = None

    def save(self):
        print("Start gathering distributed model states...")
        generator_state_dict = fsdp_state_dict(self.model.generator)
        state_dict = {"generator": generator_state_dict}
        if self.is_main_process:
            os.makedirs(os.path.join(self.output_path,
                        f"checkpoint_model_{self.step:06d}"), exist_ok=True)
            torch.save(state_dict, os.path.join(self.output_path,
                       f"checkpoint_model_{self.step:06d}", "model.pt"))
            print("Model saved to", os.path.join(self.output_path,
                  f"checkpoint_model_{self.step:06d}", "model.pt"))

    def train_one_step(self):
        self.model.eval()  # prevent dropout randomness

        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        # Step 1: text prompts + conditioning.
        batch = next(self.dataloader)
        text_prompts = batch["prompts"]
        batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size

        with torch.no_grad():
            conditional_dict = self.model.text_encoder(text_prompts=text_prompts)
            if self.unconditional_dict is None:
                unconditional_dict = self.model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * batch_size)
                unconditional_dict = {k: v.detach() for k, v in unconditional_dict.items()}
                self.unconditional_dict = unconditional_dict
            else:
                unconditional_dict = self.unconditional_dict

        # Step 2: OPD loss (rollout -> add noise -> AR vs Wan on last-21 window).
        self.generator_optimizer.zero_grad(set_to_none=True)
        opd_loss, log_dict = self.model.opd_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            text_prompts=text_prompts,
        )

        opd_loss.backward()
        generator_grad_norm = self.model.generator.clip_grad_norm_(self.max_grad_norm)
        self.generator_optimizer.step()

        # Step 3: logging.
        if self.is_main_process and not self.disable_wandb:
            wandb.log({
                "opd_loss": opd_loss.item(),
                "generator_grad_norm": generator_grad_norm.item(),
                "num_gen_frames": log_dict["num_gen_frames"].item(),
            }, step=self.step)

        # Step 4: periodic rollout visualization (every log_iters). Each rank
        # VAE-decodes its OWN on-policy rollout and saves an mp4 locally
        # (logdir/opd_vis/step{n}_rank{r}.mp4); rank 0 additionally uploads its
        # clip to wandb as a video.
        if self.step % self.config.log_iters == 0:
            self._visualize_rollout()

    def _visualize_rollout(self, num_upload_ranks=15):
        """Visualize on-policy rollouts. The first `num_upload_ranks` ranks each
        VAE-decode their OWN rollout and save an mp4 locally
        (logdir/opd_vis/step{n}_rank{r}_{F}f.mp4). After a barrier, rank 0 reads
        those files back off the shared filesystem and uploads them to wandb in a
        single batched log (one video per rank)."""
        latents = getattr(self.model, "_last_rollout_latents", None)
        num_frames = getattr(self.model, "_last_rollout_num_frames", None)
        vis_dir = os.path.join(self.output_path, "opd_vis")
        os.makedirs(vis_dir, exist_ok=True)
        rank = dist.get_rank()

        # The first `num_upload_ranks` ranks decode + save their own rollout.
        # (Guard on `latents is not None` WITHOUT early-return: every rank must
        # still reach the barrier below, or the collective will hang.)
        if latents is not None and rank < num_upload_ranks:
            local_path = os.path.join(
                vis_dir, f"step{self.step:06d}_rank{rank}_{num_frames}f.mp4")
            self.model._visualize_latents(latents, path=local_path, fps=16)

        # Wait for all producing ranks to finish writing before rank 0 reads.
        barrier()

        # rank 0 reads every produced clip back and uploads them together.
        if self.is_main_process and not self.disable_wandb:
            import imageio.v2 as imageio
            import numpy as np
            n = min(num_upload_ranks, self.world_size)
            video_logs = {}
            for r in range(n):
                p = os.path.join(vis_dir, f"step{self.step:06d}_rank{r}_{num_frames}f.mp4")
                if not os.path.exists(p):
                    continue
                try:
                    clip = np.stack(list(imageio.get_reader(p)))  # [F, H, W, C] uint8
                    video_logs[f"rollout/rank{r}"] = wandb.Video(
                        clip.transpose(0, 3, 1, 2), fps=16, format="mp4")  # [F, C, H, W]
                except Exception as e:
                    print(f"[OPD] failed to read/upload {p}: {e}")
            if video_logs:
                wandb.log(video_logs, step=self.step)

    def train(self):
        start_step = self.step
        while True:
            self.train_one_step()

            if (not self.config.no_save) and (self.step - start_step) > 0 and \
                    self.step % self.config.log_iters == 0:
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()

            self.step += 1

            barrier()
            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    if not self.disable_wandb:
                        wandb.log({"per iteration time": current_time - self.previous_time}, step=self.step)
                    self.previous_time = current_time
