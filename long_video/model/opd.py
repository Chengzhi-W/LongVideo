import torch
import torch.nn.functional as F
import torch.distributed as dist
from typing import Optional, Tuple

from model.base import SelfForcingModel


class OPD(SelfForcingModel):
    """
    On-Policy Distillation for the causal AR model.

    Motivation: the AR teacher (ar_diffusion.pt) was only trained (teacher-forcing)
    on 21 frames (5s). When it autoregressively bootstraps beyond ~21 frames it
    drifts / collapses. OPD lets a frozen bidirectional Wan teacher correct the AR
    model on the AR model's OWN on-policy rollout states:

        AR rolls out a long video (random 21/42/63 frames) -> on-policy x0 (detached)
            -> add noise -> x_t
            -> AR (causal, WITH the rollout history in its KV cache) predicts x0  (grad)
               Wan (bidirectional, frozen)                    predicts x0  (no grad, target)
            -> MSE on the LAST 21-frame window only (gradient masked on the boundary block)

    Only `generator` (the trainable causal AR model) and `real_score` (frozen
    bidirectional Wan) are used; there is no critic / fake_score.
    """

    def __init__(self, args, device):
        super().__init__(args, device)
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.num_training_frames = getattr(args, "num_training_frames", 63)

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

        self.independent_first_frame = getattr(args, "independent_first_frame", False)
        assert not self.independent_first_frame, "OPD currently assumes chunk-wise (independent_first_frame=False)"
        if self.independent_first_frame:
            self.generator.model.independent_first_frame = True
        if args.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()

        # Timestep hyperparameters for the noising step (DMD-style).
        self.num_train_timestep = args.num_train_timestep
        self.min_step = int(0.02 * self.num_train_timestep)
        self.max_step = int(0.98 * self.num_train_timestep)
        self.timestep_shift = getattr(args, "timestep_shift", 1.0)

        # CFG scale for the frozen Wan target (bidirectional). opd_use_cfg toggles it;
        # default off (conditional-only) to halve the target-side forwards.
        self.real_guidance_scale = getattr(args, "guidance_scale", 3.0)
        self.opd_use_cfg = getattr(args, "opd_use_cfg", False)

        # Multi-step diffusion rollout sampling steps (the AR model is multi-step).
        self.opd_sampling_steps = getattr(args, "opd_sampling_steps", 12)

        # Wan1.3B cache geometry, matching SelfForcingTrainingPipeline.
        self.frame_seq_length = getattr(args, "opd_frame_seq_length", 1560)
        self.num_transformer_blocks = getattr(args, "opd_num_transformer_blocks", 30)

        # Lazily created (needs the FSDP-wrapped generator).
        self.rollout_pipeline = None

    def _initialize_rollout_pipeline(self):
        """Lazy-init a diffusion inference pipeline that DRIVES the trainable
        generator (multi-step UniPC rollout). Injects the FSDP-wrapped generator /
        text_encoder / vae so no extra weights are loaded."""
        from pipeline import CausalDiffusionInferencePipeline
        # The pipeline reads args.sampling_steps; make sure it uses our OPD value.
        if getattr(self.args, "sampling_steps", None) != self.opd_sampling_steps:
            self.args.sampling_steps = self.opd_sampling_steps
        self.rollout_pipeline = CausalDiffusionInferencePipeline(
            self.args,
            device=self.device,
            generator=self.generator,
            text_encoder=self.text_encoder,
            vae=self.vae,
        )

    @torch.no_grad()
    def _rollout(self, text_prompts, image_or_video_shape) -> Tuple[torch.Tensor, int]:
        """
        Roll out a long video with the current AR model (multi-step diffusion,
        under no_grad). Length is randomly 21/42/63 frames (a whole multiple of 21),
        broadcast across ranks.

        Returns:
            - on_policy_latents: [B, num_gen_frames, C, H, W] detached clean latents.
            - num_gen_frames: int.
        """
        if self.rollout_pipeline is None:
            self._initialize_rollout_pipeline()

        B = image_or_video_shape[0]
        # Random 21/42/63 -> multiple of 21 (and of num_frame_per_block since 21 % block == 0).
        assert 21 % self.num_frame_per_block == 0
        assert self.num_training_frames % 21 == 0, \
            f"num_training_frames {self.num_training_frames} must be a multiple of 21"
        max_num_segments = self.num_training_frames // 21
        num_segments = torch.randint(1, max_num_segments + 1, (1,), device=self.device)
        # num_segments = torch.randint(3, max_num_segments + 1, (1,), device=self.device)
        dist.broadcast(num_segments, src=0)
        num_gen_frames = num_segments.item() * 21

        noise_shape = list(image_or_video_shape)
        noise_shape[1] = num_gen_frames
        noise = torch.randn(noise_shape, device=self.device, dtype=self.dtype)

        # Re-init the pipeline cache each call so it is sized to THIS rollout length
        # (rollout length varies across steps; full-history cache must fit num_gen_frames).
        self.rollout_pipeline.kv_cache_pos = None
        self.rollout_pipeline.kv_cache_neg = None
        self.rollout_pipeline.crossattn_cache_pos = None
        self.rollout_pipeline.crossattn_cache_neg = None

        on_policy_latents = self.rollout_pipeline.inference(
            noise=noise,
            text_prompts=text_prompts,
            return_video=False,
        )
        return on_policy_latents.detach().to(self.dtype), num_gen_frames

    def _sample_window_timestep(self, batch_size, num_frame) -> torch.Tensor:
        """Sample ONE DMD-style timestep for the whole 21-frame window (uniform
        across frames), with the flow-matching shift applied and clamped."""
        timestep = self._get_timestep(
            self.min_step,
            self.max_step,
            batch_size,
            num_frame,
            self.num_frame_per_block,
            uniform_timestep=True,
        )
        if self.timestep_shift > 1:
            timestep = self.timestep_shift * (timestep / 1000) / \
                (1 + (self.timestep_shift - 1) * (timestep / 1000)) * 1000
        timestep = timestep.clamp(self.min_step, self.max_step)
        return timestep

    def _empty_kv(self, batch_size, num_frames):
        """Allocate a fresh KV cache. When the AR model uses a finite local
        attention window (local_attn_size != -1, e.g. sink+21), size the cache to
        that window so the model's own eviction + sink logic triggers (train==infer
        geometry). Otherwise size it to the whole rollout (full-history, no eviction)."""
        seq = self.frame_seq_length
        local_attn_size = getattr(self.generator.model, "local_attn_size", -1)
        cache_frames = local_attn_size if local_attn_size != -1 else num_frames
        return [{
            "k": torch.zeros([batch_size, cache_frames * seq, 12, 128], dtype=self.dtype, device=self.device),
            "v": torch.zeros([batch_size, cache_frames * seq, 12, 128], dtype=self.dtype, device=self.device),
            "global_end_index": torch.tensor([0], dtype=torch.long, device=self.device),
            "local_end_index": torch.tensor([0], dtype=torch.long, device=self.device),
        } for _ in range(self.num_transformer_blocks)]

    def _empty_crossattn(self, batch_size):
        return [{
            "k": torch.zeros([batch_size, 512, 12, 128], dtype=self.dtype, device=self.device),
            "v": torch.zeros([batch_size, 512, 12, 128], dtype=self.dtype, device=self.device),
            "is_init": False,
        } for _ in range(self.num_transformer_blocks)]

    @torch.no_grad()
    def _visualize_latents(self, latents, path=None, fps=16):
        """VAE-decode clean latents [B, F, C, H, W] (first sample) to uint8 frames
        [F, H, W, C]. If `path` is given, also write an mp4 (falls back to PNGs).
        Returns the frames array so the caller can e.g. upload to wandb without
        re-decoding."""
        import numpy as np
        video = self.vae.decode_to_pixel(latents.to(self.dtype))  # [B, F, C, H, W], ~[-1,1]
        video = (video * 0.5 + 0.5).clamp(0, 1)
        frames = (video[0].permute(0, 2, 3, 1).float().cpu().numpy() * 255.0)  # [F, H, W, C]
        frames = np.clip(frames, 0, 255).astype(np.uint8)
        if path is not None:
            try:
                import imageio.v2 as imageio
                writer = imageio.get_writer(path, fps=fps, codec="libx264",
                                            quality=8, macro_block_size=None)
                for f in frames:
                    writer.append_data(f)
                writer.close()
            except Exception:
                # Fallback: dump frames as PNGs.
                import cv2, os
                os.makedirs(path + "_frames", exist_ok=True)
                for i, f in enumerate(frames):
                    cv2.imwrite(os.path.join(path + "_frames", f"{i:03d}.png"), f[:, :, ::-1])
            print(f"[OPD] visualized {frames.shape[0]} frames -> {path}")
        return frames

    def opd_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        text_prompts,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the OPD loss. See class docstring for the pipeline.
        """
        block = self.num_frame_per_block
        seq = self.frame_seq_length

        # Step 1: on-policy rollout (no_grad) -> clean long latents.
        student_full, num_gen_frames = self._rollout(text_prompts, image_or_video_shape)
        B = student_full.shape[0]
        num_contig_blocks = num_gen_frames // block
        blocks_per_window = 21 // block
        first_window_block = num_contig_blocks - blocks_per_window  # >= 0

        # Clean latents of the scored last-21 window.
        window = student_full[:, first_window_block * block:]  # [B, 21, C, H, W]

        # Step 2: sample one window timestep and noise the whole window uniformly.
        timestep = self._sample_window_timestep(B, 21)
        noise = torch.randn_like(window)
        noisy_window = self.scheduler.add_noise(
            window.flatten(0, 1),
            noise.flatten(0, 1),
            timestep.flatten(0, 1),
        ).detach().unflatten(0, (B, 21))

        # Step 3: Wan (bidirectional, frozen) target x0 on the whole noised window.
        with torch.no_grad():
            _, wan_x0_cond = self.real_score(
                noisy_image_or_video=noisy_window,
                conditional_dict=conditional_dict,
                timestep=timestep,
            )
            if self.opd_use_cfg:
                _, wan_x0_uncond = self.real_score(
                    noisy_image_or_video=noisy_window,
                    conditional_dict=unconditional_dict,
                    timestep=timestep,
                )
                wan_x0 = wan_x0_cond + (wan_x0_cond - wan_x0_uncond) * self.real_guidance_scale
            else:
                wan_x0 = wan_x0_cond
        wan_x0 = wan_x0.detach()

        # Step 4: AR (causal, trainable) predicts the window WITH rollout history.
        # Build a fresh full-history KV cache and commit the clean history
        # (frames 0..window start) under no_grad.
        kv_cache = self._empty_kv(B, num_gen_frames)
        crossattn_cache = self._empty_crossattn(B)

        def _commit(clean_block, start_frame):
            context_t = torch.zeros([B, clean_block.shape[1]], device=self.device, dtype=torch.int64)
            self.generator(
                noisy_image_or_video=clean_block,
                conditional_dict=conditional_dict,
                timestep=context_t,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=start_frame * seq,
            )

        with torch.no_grad():
            for bk in range(0, first_window_block):
                _commit(student_full[:, bk * block:(bk + 1) * block], bk * block)

        # Predict each window block WITH grad (given the clean history), then commit
        # the CLEAN block (no_grad) to advance the cache to the next position. The
        # grad-predict writes noised K/V at the block slot; the clean commit at the
        # SAME current_start overwrites that slot with clean K/V (net: cache holds
        # clean history, advanced by exactly one block).
        ar_x0 = torch.zeros_like(window)
        for j in range(blocks_per_window):
            win_block = first_window_block + j
            start_frame = win_block * block
            xt = noisy_window[:, j * block:(j + 1) * block]
            tb = timestep[:, j * block:(j + 1) * block]
            _, blk_x0 = self.generator(
                noisy_image_or_video=xt,
                conditional_dict=conditional_dict,
                timestep=tb,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=start_frame * seq,
            )
            ar_x0[:, j * block:(j + 1) * block] = blk_x0
            with torch.no_grad():
                _commit(window[:, j * block:(j + 1) * block], start_frame)

        # Step 5: MSE on the window, masking the first (boundary) block.
        gradient_mask = torch.ones_like(ar_x0, dtype=torch.bool)
        gradient_mask[:, :block] = False
        opd_loss = 0.5 * F.mse_loss(
            ar_x0.double()[gradient_mask],
            wan_x0.double()[gradient_mask],
            reduction="mean",
        )

        # Stash the on-policy rollout so the trainer can visualize it every
        # log_iters steps (VAE-decode + local save + wandb upload). Detached.
        self._last_rollout_latents = student_full
        self._last_rollout_num_frames = num_gen_frames

        log_dict = {
            "opd_loss": opd_loss.detach(),
            "num_gen_frames": torch.tensor(float(num_gen_frames), device=self.device),
            "timestep": timestep.detach(),
        }
        return opd_loss, log_dict
