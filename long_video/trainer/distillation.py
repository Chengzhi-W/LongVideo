import gc
import logging
from utils.dataset import cycle
from utils.dataset import TextDataset
from utils.distributed import EMA_FSDP, fsdp_wrap, fsdp_state_dict, launch_distributed_job, barrier
from utils.misc import set_seed
import torch.distributed as dist
from omegaconf import OmegaConf
from model import DMD
import torch
import wandb
import time
import os


class Trainer:
    def __init__(self, config):
        self.config = config
        self.step = 0

        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
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

        # use a random seed for the training
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

        # Step 2: Initialize the model and optimizer
        if config.distribution_loss == "dmd":
            self.model = DMD(config, device=self.device)
        else:
            raise ValueError("Invalid distribution matching loss")

        # Save pretrained model state_dicts to CPU
        self.fake_score_state_dict_cpu = self.model.fake_score.state_dict()

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
            wrap_strategy=config.real_score_fsdp_wrap_strategy,
            cpu_offload=False
        )

        self.model.fake_score = fsdp_wrap(
            self.model.fake_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.fake_score_fsdp_wrap_strategy,
            cpu_offload=False
        )

        # Frozen AR teacher for memory-aware DMD (only when enabled; never optimized).
        if getattr(config, "use_memory_dmd", False):
            self.model.ar_teacher = fsdp_wrap(
                self.model.ar_teacher,
                sharding_strategy=config.sharding_strategy,
                mixed_precision=config.mixed_precision,
                wrap_strategy=getattr(config, "ar_teacher_fsdp_wrap_strategy", "size"),
                cpu_offload=False
            )

        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
            cpu_offload=getattr(config, "text_encoder_cpu_offload", False)
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

        self.critic_optimizer = torch.optim.AdamW(
            [param for param in self.model.fake_score.parameters()
             if param.requires_grad],
            lr=config.lr_critic if hasattr(config, "lr_critic") else config.lr,
            betas=(config.beta1_critic, config.beta2_critic),
            weight_decay=config.weight_decay
        )

        # Step 3: Initialize the dataloader
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
            self.model.generator.load_state_dict(
                state_dict, strict=True
            )

        # Load the frozen AR teacher (memory-aware DMD).
        if getattr(config, "use_memory_dmd", False) and getattr(config, "ar_teacher_ckpt", False):
            print(f"Loading AR teacher from {config.ar_teacher_ckpt}")
            ar_sd = torch.load(config.ar_teacher_ckpt, map_location="cpu")
            for key in ("generator", "generator_ema", "model"):
                if key in ar_sd:
                    ar_sd = ar_sd[key]
                    break
            ar_sd = {k.replace("model._fsdp_wrapped_module.", "model.", 1): v for k, v in ar_sd.items()}
            self.model.ar_teacher.load_state_dict(ar_sd, strict=True)

        ##############################################################################################################

        # Let's delete EMA params for early steps to save some computes at training and inference
        if self.step < config.ema_start_step:
            self.generator_ema = None

        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 10.0)
        self.max_grad_norm_critic = getattr(config, "max_grad_norm_critic", 10.0)
        self.previous_time = None

    def save(self):
        print("Start gathering distributed model states...")
        generator_state_dict = fsdp_state_dict(
            self.model.generator)
        critic_state_dict = fsdp_state_dict(
            self.model.fake_score)

        if self.config.ema_start_step < self.step:
            state_dict = {
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

    def save_critic(self):
        print("Start gathering distributed model states...")
        
        critic_state_dict = fsdp_state_dict(
            self.model.fake_score)

        
        state_dict = critic_state_dict

        if self.is_main_process:
            os.makedirs(os.path.join(self.output_path,
                        f"checkpoint_model_{self.step:06d}"), exist_ok=True)
            torch.save(state_dict, os.path.join(self.output_path,
                       f"checkpoint_model_{self.step:06d}", "model.pt"))
            print("Model saved to", os.path.join(self.output_path,
                  f"checkpoint_model_{self.step:06d}", "model.pt"))

    @torch.no_grad()
    def _save_latents_mp4(self, latents, path, fps=16):
        """VAE-decode clean latents [B, F, C, H, W] (first sample) and write an mp4
        (falls back to per-frame PNGs if the video writer is unavailable)."""
        import numpy as np
        video = self.model.vae.decode_to_pixel(latents.to(self.dtype))  # [B, F, C, H, W] ~[-1,1]
        video = (video * 0.5 + 0.5).clamp(0, 1)
        frames = (video[0].permute(0, 2, 3, 1).float().cpu().numpy() * 255.0)
        frames = np.clip(frames, 0, 255).astype(np.uint8)  # [F, H, W, C]
        try:
            import imageio.v2 as imageio
            writer = imageio.get_writer(path, fps=fps, codec="libx264",
                                        quality=8, macro_block_size=None)
            for f in frames:
                writer.append_data(f)
            writer.close()
        except Exception:
            import cv2
            os.makedirs(path + "_frames", exist_ok=True)
            for i, f in enumerate(frames):
                cv2.imwrite(os.path.join(path + "_frames", f"{i:03d}.png"), f[:, :, ::-1])
        print(f"[vis] rollout ({frames.shape[0]} frames) -> {path}")

    @torch.no_grad()
    def _visualize_demo(self):
        """Comparable demo visualization: every `vis_iters` steps, generate a
        FULL-length (num_training_frames) video for a FIXED prompt using a FIXED init
        noise and FULL denoising (last_step_only), then upload to wandb. Same prompt +
        noise every step and across runs -> directly comparable over training.

        Distributed: the pipeline runs the FSDP generator and calls dist.broadcast
        internally, so EVERY rank must execute the pipeline call in lockstep or the
        collectives deadlock. We therefore have every rank render prompt (rank %
        n_prompts) with the SAME per-prompt noise seed, so ranks that share a prompt
        produce identical clips; rank 0 then uploads one clip per distinct prompt.
        `inference_pipeline` is lazily initialized during training rollouts, which all
        ranks run together, so `pipe is None` is consistent across ranks (no partial
        participation)."""
        rank = dist.get_rank()

        # Cache the full prompt pool once; RE-SAMPLE k prompts every eval so we see
        # diversity (a different batch each step). Seed = vis_seed + step, so the same
        # step still picks the SAME prompts across runs (cross-run comparability kept;
        # only cross-step comparability is dropped). Set vis_fixed_prompts: true to
        # freeze the same k prompts across all steps instead.
        allp = getattr(self, "_vis_prompt_pool", None)
        if allp is None:
            path = getattr(self.config, "vis_prompt_path", "prompts/demos.txt")
            with open(path) as f:
                allp = [l.strip() for l in f if l.strip()]
            self._vis_prompt_pool = allp
        k = int(getattr(self.config, "vis_num_prompts", 8))
        base_seed = int(getattr(self.config, "vis_seed", 0))
        fixed = bool(getattr(self.config, "vis_fixed_prompts", False))
        sample_seed = base_seed if fixed else base_seed + self.step
        g = torch.Generator().manual_seed(sample_seed)
        idx = torch.randperm(len(allp), generator=g)[:k].tolist()
        prompts = [allp[i] for i in idx]
        n_prompts = len(prompts)

        pipe = getattr(self.model, "inference_pipeline", None)
        if pipe is None:
            # Pipeline not built yet (only before the first training rollout). All
            # ranks see None together, so returning here keeps them in lockstep.
            return

        # Which prompt THIS rank renders (all ranks participate -> collectives stay
        # synced). Prompt index is deterministic from rank, seeded noise is keyed to
        # the prompt index so duplicate renders across ranks are identical.
        p_idx = rank % n_prompts
        my_prompt = prompts[p_idx]
        shape = list(self.config.image_or_video_shape)
        shape[0] = 1
        shape[1] = self.model.num_training_frames  # full length (63 = 15s)
        gen = torch.Generator(device=self.device).manual_seed(
            1234 + p_idx + int(getattr(self.config, "vis_seed", 0)))
        noise = torch.randn(shape, generator=gen, device=self.device, dtype=self.dtype)
        cond = self.model.text_encoder(text_prompts=[my_prompt])

        prev_lso = pipe.last_step_only
        pipe.last_step_only = True  # full denoising -> clean deterministic output
        try:
            out = pipe.inference_with_trajectory(noise=noise, **cond)
        finally:
            pipe.last_step_only = prev_lso
        latents = out[0] if isinstance(out, tuple) else out

        # Only the FIRST rank that owns each prompt writes it (avoids duplicate writes
        # from ranks sharing a prompt); the VAE decode is local (no collectives).
        if rank < n_prompts:
            demo_dir = os.path.join(self.output_path, "demo_vis", f"prompt{p_idx}")
            os.makedirs(demo_dir, exist_ok=True)
            self._save_latents_mp4(latents, os.path.join(demo_dir, f"step{self.step:06d}.mp4"), fps=16)

        barrier()

        if self.is_main_process and not self.disable_wandb:
            video_logs = {}
            for r in range(n_prompts):
                p = os.path.join(self.output_path, "demo_vis", f"prompt{r}", f"step{self.step:06d}.mp4")
                if not os.path.exists(p):
                    continue
                try:
                    video_logs[f"demo/prompt{r}"] = wandb.Video(p, fps=16, format="mp4")
                except Exception as e:
                    print(f"[demo-vis] failed to upload {p}: {e}")
            if video_logs:
                wandb.log(video_logs, step=self.step)

    def fwdbwd_one_step(self, batch, train_generator, clean_latent=None):
        self.model.eval()  # prevent any randomness (e.g. dropout)

        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        # Step 1: Get the next batch of text prompts
        text_prompts = batch["prompts"]
        if self.config.i2v:
            # clean_latent = None #original code here
            image_latent = batch["ode_latent"][:, -1][:, 0:1, ].to(
                device=self.device, dtype=self.dtype)
        else:
            # clean_latent = None #original code here
            image_latent = None

        batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size

        # Step 2: Extract the conditional infos
        with torch.no_grad():
            conditional_dict = self.model.text_encoder(
                text_prompts=text_prompts)

            if not getattr(self, "unconditional_dict", None):
                unconditional_dict = self.model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * batch_size)
                unconditional_dict = {k: v.detach()
                                      for k, v in unconditional_dict.items()}
                self.unconditional_dict = unconditional_dict  # cache the unconditional_dict
            else:
                unconditional_dict = self.unconditional_dict

        # Step 3: Store gradients for the generator (if training the generator)
        if train_generator:
            generator_loss, generator_log_dict = self.model.generator_loss(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=clean_latent,
                initial_latent=image_latent if self.config.i2v else None
            )
           
            generator_loss.backward()
            generator_grad_norm = self.model.generator.clip_grad_norm_(
                self.max_grad_norm_generator)

            generator_log_dict.update({"generator_loss": generator_loss,
                                       "generator_grad_norm": generator_grad_norm})

            return generator_log_dict
        else:
            generator_log_dict = {}

        # Step 4: Store gradients for the critic (if training the critic)
        critic_loss, critic_log_dict = self.model.critic_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent,
            initial_latent=image_latent if self.config.i2v else None
        )

        critic_loss.backward()
        critic_grad_norm = self.model.fake_score.clip_grad_norm_(
            self.max_grad_norm_critic)

        critic_log_dict.update({"critic_loss": critic_loss,
                                "critic_grad_norm": critic_grad_norm})

        return critic_log_dict


    def train(self):
        start_step = self.step
       
        while True:
            TRAIN_GENERATOR = self.step % self.config.dfake_gen_update_ratio == 0

            # Train the generator
            if TRAIN_GENERATOR:
                self.generator_optimizer.zero_grad(set_to_none=True)

                batch = next(self.dataloader)
                generator_log_dict = self.fwdbwd_one_step(batch, True)

                self.generator_optimizer.step()
                if self.generator_ema is not None:
                    self.generator_ema.update(self.model.generator)
                
                
                

            # Train the critic
            self.critic_optimizer.zero_grad(set_to_none=True)
            batch = next(self.dataloader)
            critic_log_dict = self.fwdbwd_one_step(batch, False)
                
            self.critic_optimizer.step()

            # Increment the step since we finished gradient update
            self.step += 1

            # Create EMA params (if not already created)
            if (self.step >= self.config.ema_start_step) and \
                    (self.generator_ema is None) and (self.config.ema_weight > 0):
                self.generator_ema = EMA_FSDP(self.model.generator, decay=self.config.ema_weight)

            # Save the model
            if (not self.config.no_save) and (self.step - start_step) > 0 and self.step % self.config.log_iters == 0:
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()

            # Comparable fixed-prompt demo videos (every vis_iters), full-length + full
            # denoising, so quality is directly comparable across steps and runs.
            vis_iters = int(getattr(self.config, "vis_iters", 0) or 0)
            if (not self.config.no_visualize) and vis_iters > 0 and self.step % vis_iters == 0:
                self._visualize_demo()

            # Logging
            if self.is_main_process:
                wandb_loss_dict = {}
                if TRAIN_GENERATOR:
                    wandb_loss_dict.update(
                        {
                            "generator_loss": generator_log_dict["generator_loss"].mean().item(),
                            "generator_grad_norm": generator_log_dict["generator_grad_norm"].mean().item(),
                            "dmdtrain_gradient_norm": generator_log_dict["dmdtrain_gradient_norm"].mean().item()
                        }
                    )
                    if "mem_grad_norm" in generator_log_dict:
                        wandb_loss_dict["mem_grad_norm"] = generator_log_dict["mem_grad_norm"].mean().item()
                    if "bi_grad_norm" in generator_log_dict:
                        wandb_loss_dict["bi_grad_norm"] = generator_log_dict["bi_grad_norm"].mean().item()

                wandb_loss_dict.update(
                    {
                        "critic_loss": critic_log_dict["critic_loss"].mean().item(),
                        "critic_grad_norm": critic_log_dict["critic_grad_norm"].mean().item()
                    }
                )

                if not self.disable_wandb:
                    wandb.log(wandb_loss_dict, step=self.step)

            if self.step % self.config.gc_interval == 0:
                if dist.get_rank() == 0:
                    logging.info("DistGarbageCollector: Running GC.")
                gc.collect()
                torch.cuda.empty_cache()

            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    if not self.disable_wandb:
                        wandb.log({"per iteration time": current_time - self.previous_time}, step=self.step)
                    self.previous_time = current_time
