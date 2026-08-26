from typing import Tuple
import torch

from model.base import BaseModel
from utils.teacher_forcing import (
    build_balanced_target_block_patterns,
    expand_target_blocks,
)
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper


class CausalDiffusion(BaseModel):
    def __init__(self, args, device):
        """
        Initialize the Diffusion loss module.
        """
        super().__init__(args, device)
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block
        self.independent_first_frame = getattr(args, "independent_first_frame", False)
        if self.independent_first_frame:
            self.generator.model.independent_first_frame = True

        if args.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()

        # Step 2: Initialize all hyperparameters
        self.num_train_timestep = args.num_train_timestep
        self.min_step = int(0.02 * self.num_train_timestep)
        self.max_step = int(0.98 * self.num_train_timestep)
        self.guidance_scale = args.guidance_scale
        self.timestep_shift = getattr(args, "timestep_shift", 1.0)
        self.teacher_forcing = getattr(args, "teacher_forcing", False)

        # Sparse-target teacher forcing keeps the complete clean 20-second
        # history but only packs a balanced subset of noisy current chunks.
        # A value of zero preserves the original all-target training path.
        self.sparse_target_num_blocks = int(
            getattr(args, "sparse_target_num_blocks", 0))
        self.sparse_target_num_patterns = int(
            getattr(args, "sparse_target_num_patterns", 9))
        self.sparse_target_pattern_stride = int(
            getattr(args, "sparse_target_pattern_stride", 2))
        self._sparse_target_pattern_cache = {}
        if self.sparse_target_num_blocks < 0:
            raise ValueError("sparse_target_num_blocks must be non-negative")
        if self.sparse_target_num_blocks and not self.teacher_forcing:
            raise ValueError(
                "sparse_target_num_blocks requires teacher_forcing=true")
        
        # Noise augmentation in teacher forcing, we add small noise to clean context latents
        self.noise_augmentation_max_timestep = getattr(args, "noise_augmentation_max_timestep", 0)

    def _sample_target_frame_indices(
        self, num_frames: int, device: torch.device
    ) -> torch.Tensor | None:
        if self.sparse_target_num_blocks == 0:
            return None

        cache_key = (num_frames, self.num_frame_per_block)
        patterns = self._sparse_target_pattern_cache.get(cache_key)
        if patterns is None:
            patterns = build_balanced_target_block_patterns(
                num_frames=num_frames,
                block_size=self.num_frame_per_block,
                num_target_blocks=self.sparse_target_num_blocks,
                num_patterns=self.sparse_target_num_patterns,
                traversal_stride=self.sparse_target_pattern_stride,
            )
            self._sparse_target_pattern_cache[cache_key] = patterns

        pattern_index = int(torch.randint(
            len(patterns), (1,), device=device).item())
        target_frames = expand_target_blocks(
            patterns[pattern_index],
            num_frames=num_frames,
            block_size=self.num_frame_per_block,
        )
        return torch.tensor(target_frames, device=device, dtype=torch.long)

    def _initialize_models(self, args, device):
        self.generator = WanDiffusionWrapper(**getattr(args, "model_kwargs", {}), is_causal=True)
        self.generator.model.requires_grad_(True)

        self.text_encoder = WanTextEncoder()
        self.text_encoder.requires_grad_(False)

        self.vae = WanVAEWrapper()
        self.vae.requires_grad_(False)
        
        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noise and compute the DMD loss.
        The noisy input to the generator is backward simulated.
        This removes the need of any datasets during distillation.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
        Output:
            - loss: a scalar tensor representing the generator loss.
            - generator_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        batch_size, num_frame = clean_latent.shape[:2]
        target_frame_indices = self._sample_target_frame_indices(
            num_frames=num_frame, device=clean_latent.device)
        if target_frame_indices is None:
            target_clean_latent = clean_latent
        else:
            target_clean_latent = clean_latent.index_select(
                1, target_frame_indices)

        target_num_frames = target_clean_latent.shape[1]
        target_shape = [batch_size, target_num_frames, *clean_latent.shape[2:]]
        noise = torch.randn_like(target_clean_latent)

        # Step 2: Randomly sample a timestep and add noise to denoiser inputs
        index = self._get_timestep(
            0,
            self.scheduler.num_train_timesteps,
            target_shape[0],
            target_shape[1],
            self.num_frame_per_block,
            uniform_timestep=False
        )
        timestep = self.scheduler.timesteps[index].to(dtype=self.dtype, device=self.device)
        noisy_latents = self.scheduler.add_noise(
            target_clean_latent.flatten(0, 1),
            noise.flatten(0, 1),
            timestep.flatten(0, 1)
        ).unflatten(0, (batch_size, target_num_frames))
        training_target = self.scheduler.training_target(
            target_clean_latent, noise, timestep)

        # Step 3: Noise augmentation, also add small noise to clean context latents
        if self.noise_augmentation_max_timestep > 0:
            index_clean_aug = self._get_timestep(
                self.noise_augmentation_max_timestep,
                1000,
                batch_size,
                num_frame,
                self.num_frame_per_block,
                uniform_timestep=False
            )
            timestep_clean_aug = self.scheduler.timesteps[index_clean_aug].to(dtype=self.dtype, device=self.device)
            clean_noise = torch.randn_like(clean_latent)
            clean_latent_aug = self.scheduler.add_noise(
                clean_latent.flatten(0, 1),
                clean_noise.flatten(0, 1),
                timestep_clean_aug.flatten(0, 1)
            ).unflatten(0, (batch_size, num_frame))
        else:
            clean_latent_aug = clean_latent
            timestep_clean_aug = None
        # Compute loss
        
        
        flow_pred, x0_pred = self.generator(
            noisy_image_or_video=noisy_latents,
            conditional_dict=conditional_dict,
            timestep=timestep,
            clean_x=clean_latent_aug if self.teacher_forcing else None,
            aug_t=timestep_clean_aug if self.teacher_forcing else None,
            teacher_forcing_target_frame_indices=target_frame_indices,
        )
        # loss = torch.nn.functional.mse_loss(flow_pred.float(), training_target.float())
        loss = torch.nn.functional.mse_loss(
            flow_pred.float(), training_target.float(), reduction='none'
        ).mean(dim=(2, 3, 4))
        loss = loss * self.scheduler.training_weight(timestep).unflatten(
            0, (batch_size, target_num_frames))
        loss = loss.mean()

        log_dict = {
            "x0": target_clean_latent.detach(),
            "x0_pred": x0_pred.detach()
        }
        return loss, log_dict
