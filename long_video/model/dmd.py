from pipeline import SelfForcingTrainingPipeline
import torch.nn.functional as F
from typing import Optional, Tuple
import torch
import torch.distributed as dist

from model.base import SelfForcingModel
from utils.wan_wrapper import WanDiffusionWrapper


class DMD(SelfForcingModel):
    def __init__(self, args, device):
        """
        Initialize the DMD (Distribution Matching Distillation) module.
        This class is self-contained and compute generator and fake score losses
        in the forward pass.
        """
        super().__init__(args, device)
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.same_step_across_blocks = getattr(args, "same_step_across_blocks", True)
        self.num_training_frames = getattr(args, "num_training_frames", 21)

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

        self.independent_first_frame = getattr(args, "independent_first_frame", False)
        if self.independent_first_frame:
            self.generator.model.independent_first_frame = True
        if args.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()
            self.fake_score.enable_gradient_checkpointing()

        # this will be init later with fsdp-wrapped modules
        self.inference_pipeline: SelfForcingTrainingPipeline = None

        # Step 2: Initialize all dmd hyperparameters
        self.num_train_timestep = args.num_train_timestep
        self.min_step = int(0.02 * self.num_train_timestep)
        self.max_step = int(0.98 * self.num_train_timestep)
        if hasattr(args, "real_guidance_scale"):
            self.real_guidance_scale = args.real_guidance_scale
            self.fake_guidance_scale = args.fake_guidance_scale
        else:
            self.real_guidance_scale = args.guidance_scale
            self.fake_guidance_scale = 0.0
        self.timestep_shift = getattr(args, "timestep_shift", 1.0)
        self.ts_schedule = getattr(args, "ts_schedule", True)
        self.ts_schedule_max = getattr(args, "ts_schedule_max", False)
        self.min_score_timestep = getattr(args, "min_score_timestep", 0)

        # Memory-aware DMD: augment the Wan real score with the AR teacher's memory
        # effect dD_A = x0_AR(x_t|h) - x0_AR(x_t|empty). `self.use_memory_dmd` /
        # `self.ar_teacher` are set in BaseModel. Default off -> vanilla DMD.
        if getattr(self, "use_memory_dmd", False):
            self.memory_dmd_scale = float(getattr(args, "memory_dmd_scale", 1.0))
            self.ar_frame_seq_length = getattr(args, "ar_frame_seq_length", 1560)
            self.ar_num_transformer_blocks = getattr(args, "ar_num_transformer_blocks", 30)
            # If True, high-pass the memory signal (subtract per-frame/channel spatial
            # mean) so it teaches only structural long-range memory, not the global
            # colour/saturation shift the student already learns from plain DMD.
            self.memory_highpass = bool(getattr(args, "memory_highpass", False))
            # Re-based window layout: sink frames (placed at re-based pos [0, n_sink-1]).
            # recent = 21 - n_sink - block fills the rest before the noisy chunk.
            self.memory_sink_frames = int(getattr(args, "memory_sink_frames", 3))
            # Timestep gating: only apply the memory term when the DMD noise level
            # (timestep) is BELOW this threshold. Rationale: high-t (high-noise) steps
            # decide global structure & large motion; low-t (low-noise) steps refine
            # detail & temporal consistency. Restricting memory to low-t lets it keep
            # the consistency gain while leaving motion decisions (high-t) untouched,
            # to curb the dynamic-degree drop. Default 10**9 = no gating (all timesteps).
            self.memory_max_timestep = float(getattr(args, "memory_max_timestep", 1e9))

        if getattr(self.scheduler, "alphas_cumprod", None) is not None:
            self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
        else:
            self.scheduler.alphas_cumprod = None

    def _ar_empty_cache(self, batch_size, num_frames):
        """Fresh (kv_cache, crossattn_cache) for the AR teacher. Sized to the local
        attention window if finite (eviction+sink geometry), else the whole rollout."""
        seq = self.ar_frame_seq_length
        local = getattr(self.ar_teacher.model, "local_attn_size", -1)
        cf = local if local != -1 else num_frames
        kv = [{
            "k": torch.zeros([batch_size, cf * seq, 12, 128], dtype=self.dtype, device=self.device),
            "v": torch.zeros([batch_size, cf * seq, 12, 128], dtype=self.dtype, device=self.device),
            "global_end_index": torch.tensor([0], dtype=torch.long, device=self.device),
            "local_end_index": torch.tensor([0], dtype=torch.long, device=self.device),
        } for _ in range(self.ar_num_transformer_blocks)]
        cross = [{
            "k": torch.zeros([batch_size, 512, 12, 128], dtype=self.dtype, device=self.device),
            "v": torch.zeros([batch_size, 512, 12, 128], dtype=self.dtype, device=self.device),
            "is_init": False,
        } for _ in range(self.ar_num_transformer_blocks)]
        return kv, cross

    def _ar_commit(self, clean_block, kv, cross, start_frame, conditional_dict):
        """Commit a clean block (timestep 0) into the AR cache at absolute frame pos."""
        B, blk = clean_block.shape[:2]
        self.ar_teacher(
            noisy_image_or_video=clean_block, conditional_dict=conditional_dict,
            timestep=torch.zeros([B, blk], device=self.device, dtype=torch.int64),
            kv_cache=kv, crossattn_cache=cross,
            current_start=start_frame * self.ar_frame_seq_length,
        )

    @torch.no_grad()
    def _memory_effect(self, student_full, x_t, timestep, conditional_dict):
        """AR-teacher memory effect on ONE random non-boundary chunk of the scored
        window, with POSITION RE-BASING to stay inside the teacher's trained RoPE range:

            dD_A = x0_AR(x_t | sink+recent-history) - x0_AR(x_t | empty)

        Re-basing (the point of this experiment): the AR teacher was only trained on
        21-frame absolute positions [0,20]. Committing history at absolute frame
        42..60 is RoPE-OOD, so the hist-nohist differential is untrustworthy. Instead
        we build a SINGLE re-based 21-frame window laid out CONTIGUOUSLY:

            sink   (first `n_sink` frames of the rollout)   -> re-based pos [0, s-1]
            recent (the `n_recent` frames before the chunk) -> re-based pos [s, s+r-1]
            noisy chunk (block frames)                       -> re-based pos [s+r, 20]

        with n_sink + n_recent + block = 21, so every position stays in [0,20] (no
        extrapolation, no eviction). This isolates whether the memory signal is dirty
        purely because of position OOD (fixed here) rather than a teacher-capability
        gap. Both branches (hist=real frames, nohist=zeros) commit the SAME count at
        the SAME re-based positions, so the diff isolates history CONTENT.

        Input:
            - student_full: contiguous clean rollout [B, F_full, C, H, W].
            - x_t: the DMD-noised scored window [B, 21, C, H, W] (same x_t Wan sees).
            - timestep: [B, 21] the window timestep.
        Output:
            - memory: [B, 21, C, H, W]; non-zero only on the chosen chunk.
        """
        block = self.num_frame_per_block
        B, num_frames_full = student_full.shape[:2]
        seq = self.ar_frame_seq_length
        num_blocks_full = num_frames_full // block
        blocks_per_window = 21 // block
        first_window_block = num_blocks_full - blocks_per_window  # >=0

        # Re-based 21-frame window layout: sink + recent + noisy-chunk = 21.
        n_sink = int(getattr(self, "memory_sink_frames", 3))
        n_recent = 21 - n_sink - block                 # e.g. 21 - 3 - 3 = 15

        # Pick ONE random non-boundary chunk of the scored window (j in [1, K-1]).
        # Broadcast so every rank runs the same commit/predict sequence (ar_teacher is
        # FSDP-wrapped -> each forward is a collective; a per-rank j would desync).
        j = torch.randint(1, blocks_per_window, (1,), device=self.device)
        dist.broadcast(j, src=0)
        j = j.item()

        chunk_abs_start = (first_window_block + j) * block   # first frame of the chunk
        # `recent` = the n_recent frames immediately before the chunk (clamped to 0).
        rec_start = max(0, chunk_abs_start - n_recent)
        recent = student_full[:, rec_start:chunk_abs_start]  # [B, r, C, H, W], r<=n_recent
        r = recent.shape[1]
        # `sink` = the first n_sink frames of the rollout; clamp so it never overlaps
        # the recent slice (short rollouts).
        sink_end = min(n_sink, rec_start)
        sink = student_full[:, 0:sink_end]                   # [B, s, C, H, W], s<=n_sink
        s = sink.shape[1]

        memory = torch.zeros_like(x_t)
        xt_blk = x_t[:, j * block:(j + 1) * block]
        tb = timestep[:, j * block:(j + 1) * block]
        preds = {}
        for tag, use_hist in (("hist", True), ("nohist", False)):
            kv, cross = self._ar_empty_cache(B, 21)
            if use_hist:
                # Commit sink at re-based pos [0..s-1], then recent at [s..s+r-1], and
                # predict the chunk right after, at re-based pos [s+r .. s+r+block-1].
                for src_full, base in ((sink, 0), (recent, s)):
                    for c in range(0, src_full.shape[1], block):
                        blk = src_full[:, c:c + block]
                        self._ar_commit(blk, kv, cross, base + c, conditional_dict)
                chunk_pos = s + r
            else:
                # nohist = the teacher's genuine NO-HISTORY prior: empty cache, predict
                # the chunk as if it were the FIRST block of a fresh rollout (pos 0).
                # This is in-distribution for the teacher (every rollout starts this
                # way), unlike committing zero/black frames as fake history, which is
                # OOD garbage and would poison the differential.
                chunk_pos = 0
            _, x0 = self.ar_teacher(
                noisy_image_or_video=xt_blk, conditional_dict=conditional_dict,
                timestep=tb, kv_cache=kv, crossattn_cache=cross,
                current_start=chunk_pos * seq,
            )
            preds[tag] = x0
        memory[:, j * block:(j + 1) * block] = preds["hist"] - preds["nohist"]

        # Optional high-pass: strip each frame/channel's spatial mean from the memory
        # signal. The hist-vs-nohist difference is dominated by a large, low-frequency
        # colour/brightness/saturation shift (which the student already learns from
        # plain DMD); subtracting the per-(frame,channel) spatial mean removes that
        # global tint and keeps only the STRUCTURAL difference (where objects are, e.g.
        # the moon / skyline), which is the long-range memory the student can't get on
        # its own. mean over (H, W) per [B, F, C].
        if getattr(self, "memory_highpass", False):
            spatial_mean = memory.mean(dim=(3, 4), keepdim=True)
            memory = memory - spatial_mean

        return memory.detach()

    def _compute_kl_grad(
        self, noisy_image_or_video: torch.Tensor,
        estimated_clean_image_or_video: torch.Tensor,
        timestep: torch.Tensor,
        conditional_dict: dict, unconditional_dict: dict,
        normalization: bool = True,
        student_full_latent: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the KL grad (eq 7 in https://arxiv.org/abs/2311.18828), in x0-space.
        The fake and real scores are both taken as the model's x0 prediction, so
        grad = x0_fake - x0_real; the caller's loss is likewise in x0-space (see
        compute_distribution_matching_loss).
        Input:
            - noisy_image_or_video: a tensor with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - estimated_clean_image_or_video: a tensor with shape [B, F, C, H, W] representing the estimated clean image or video (x0).
            - timestep: a tensor with shape [B, F] containing the randomly generated timestep.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - normalization: a boolean indicating whether to normalize the gradient.
        Output:
            - kl_grad: a tensor representing the KL grad (x0-space).
            - kl_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        # Step 1: Compute the fake score (x0-space).
        _, pred_fake_image_cond = self.fake_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=conditional_dict,
            timestep=timestep
        )

        if self.fake_guidance_scale != 0.0:
            _, pred_fake_image_uncond = self.fake_score(
                noisy_image_or_video=noisy_image_or_video,
                conditional_dict=unconditional_dict,
                timestep=timestep
            )
            pred_fake_image = pred_fake_image_cond + (
                pred_fake_image_cond - pred_fake_image_uncond
            ) * self.fake_guidance_scale
        else:
            pred_fake_image = pred_fake_image_cond

        # Step 2: Compute the real score (bidirectional teacher) at the SAME x_t,
        # in x0-space. Conditional + unconditional -> cfg (https://arxiv.org/abs/2207.12598)
        _, pred_real_image_cond = self.real_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=conditional_dict,
            timestep=timestep
        )

        _, pred_real_image_uncond = self.real_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=unconditional_dict,
            timestep=timestep
        )

        pred_real_image = pred_real_image_cond + (
            pred_real_image_cond - pred_real_image_uncond
        ) * self.real_guidance_scale

        # Step 2b: memory-aware real target. AR teacher adds only the memory effect
        # dD_A = x0_AR(x_t|h) - x0_AR(x_t|empty) on one randomly chosen chunk.
        # DUAL normalization (do NOT fold memory into pred_real_image): the DMD term
        # keeps its own n_dmd on the pure Wan target, and memory gets its OWN scale
        # N_mem = E|memory| over the (sparse) non-zero chunk. Folding it in shared a
        # denominator that -> 0 as x0 -> pred_real (late-training memory blow-up) and
        # diluted the sparse 3/21-frame signal across the whole window.
        # Requires a rollout LONGER than one 21-frame window: the scored last window
        # needs real frames before it to serve as the AR teacher's history. A 5s
        # (exactly 21-frame) rollout has none, so the memory term is skipped there.
        memory = None
        if student_full_latent is not None and student_full_latent.shape[1] > 21:
            memory = self._memory_effect(
                student_full_latent, noisy_image_or_video, timestep, conditional_dict)
            # Timestep gating (per-sample): zero out the memory term when the DMD noise
            # level is at/above memory_max_timestep, so memory only refines low-noise
            # (detail/consistency) steps and leaves high-noise (motion-deciding) steps
            # alone. We mask the RESULT rather than skipping _memory_effect, because
            # timestep is sampled PER RANK (not broadcast); skipping would make ranks
            # issue different numbers of FSDP ar_teacher collectives -> NCCL deadlock.
            # All ranks always run the full ar_teacher forward; only the gate mask differs.
            if self.memory_max_timestep < 1e9:
                # timestep is [B, F] uniform across the window; reduce to [B] then broadcast.
                t_per_sample = timestep.float().mean(dim=1)                  # [B]
                gate = (t_per_sample < self.memory_max_timestep).float()     # [B], 1=apply
                memory = memory * gate.view(-1, 1, 1, 1, 1)

        # Step 3: build the x0-space DMD gradient (original DMD eq.8 form):
        #   grad = (pred_fake - pred_real) / n_dmd,   n_dmd = E|x0 - pred_real|
        dims = [1, 2, 3, 4]
        eps = 1e-6
        if normalization:
            n_dmd = (estimated_clean_image_or_video - pred_real_image).abs().mean(
                dim=dims, keepdim=True).detach().clamp_min(eps)
        else:
            n_dmd = torch.ones([], device=noisy_image_or_video.device)

        grad = (pred_fake_image - pred_real_image) / n_dmd

        # Norm of the pure bidirectional (DMD) term, BEFORE the memory correction.
        # Together with mem_grad_norm this decomposes the final gradient into its two
        # sources (bi vs memory). NOTE they are the norms of two separate ADDENDS
        # (grad = bi_term - mem_term), so |final grad| != bi + mem in general.
        bi_grad_norm = torch.mean(torch.abs(grad)).detach()

        # Add the memory term WITHOUT its own normalization: grad = bi_term - lambda*memory.
        # memory = x0_AR(hist) - x0_AR(nohist) is already an x0-space residual (same
        # space as the DMD numerator), so we let its RAW magnitude through and let
        # lambda be the single knob for its strength. Dividing by N_mem=E|memory|
        # (the previous behaviour) erased the per-chunk strength of the history effect
        # (weak-history chunks got pulled just as hard as strong ones), which
        # over-suppressed motion; and it never touches n_dmd, so there is no blow-up.
        # NOTE: lambda's useful range differs from the normalized version -- re-sweep it.
        mem_grad_norm = torch.zeros([], device=noisy_image_or_video.device)
        if memory is not None:
            mem_term = self.memory_dmd_scale * memory
            grad = grad - mem_term
            mem_grad_norm = torch.mean(torch.abs(mem_term)).detach()

        grad = torch.nan_to_num(grad)

        return grad, {
            "dmdtrain_gradient_norm": torch.mean(torch.abs(grad)).detach(),
            "bi_grad_norm": bi_grad_norm,
            "mem_grad_norm": mem_grad_norm,
            "timestep": timestep.detach(),
        }

    def compute_distribution_matching_loss(
        self,
        image_or_video: torch.Tensor,
        conditional_dict: dict,
        unconditional_dict: dict,
        gradient_mask: Optional[torch.Tensor] = None,
        denoised_timestep_from: int = 0,
        denoised_timestep_to: int = 0,
        student_full_latent: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the DMD loss (eq 7 in https://arxiv.org/abs/2311.18828).
        Input:
            - image_or_video: a tensor with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - gradient_mask: a boolean tensor with the same shape as image_or_video indicating which pixels to compute loss .
        Output:
            - dmd_loss: a scalar tensor representing the DMD loss.
            - dmd_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        original_latent = image_or_video

        batch_size, num_frame = image_or_video.shape[:2]
        with torch.no_grad():
            # Step 1: Randomly sample timestep based on the given schedule and corresponding noise
            min_timestep = denoised_timestep_to if self.ts_schedule and denoised_timestep_to is not None else self.min_score_timestep
            max_timestep = denoised_timestep_from if self.ts_schedule_max and denoised_timestep_from is not None else self.num_train_timestep
            timestep = self._get_timestep(
                min_timestep,
                max_timestep,
                batch_size,
                num_frame,
                self.num_frame_per_block,
                uniform_timestep=True
            )

            # TODO:should we change it to `timestep = self.scheduler.timesteps[timestep]`?
            if self.timestep_shift > 1:
                timestep = self.timestep_shift * \
                    (timestep / 1000) / \
                    (1 + (self.timestep_shift - 1) * (timestep / 1000)) * 1000
            timestep = timestep.clamp(self.min_step, self.max_step)

            noise = torch.randn_like(image_or_video)
            noisy_latent = self.scheduler.add_noise(
                image_or_video.flatten(0, 1),
                noise.flatten(0, 1),
                timestep.flatten(0, 1)
            ).detach().unflatten(0, (batch_size, num_frame))

            # Step 2: Compute the KL grad (x0-space).
            grad, dmd_log_dict = self._compute_kl_grad(
                noisy_image_or_video=noisy_latent,
                estimated_clean_image_or_video=original_latent,
                timestep=timestep,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                student_full_latent=student_full_latent,
            )

        # The DMD loss is expressed in x0-space (original DMD recipe): the student's
        # x0 (original_latent) regresses onto the detached target (x0 - grad), where
        # grad = x0_fake - x0_real. Minimising this MSE pushes x0 along the negative
        # DMD gradient.
        if gradient_mask is not None:
            # Useless if we set always 21 latent frames
            dmd_loss = 0.5 * F.mse_loss(original_latent.double(
            )[gradient_mask], (original_latent.double() - grad.double()).detach()[gradient_mask], reduction="mean")
        else:
            dmd_loss = 0.5 * F.mse_loss(original_latent.double(
            ), (original_latent.double() - grad.double()).detach(), reduction="mean")
        return dmd_loss, dmd_log_dict

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
        # Step 1: Unroll generator to obtain fake videos
        pred_image, gradient_mask, denoised_timestep_from, denoised_timestep_to = self._run_generator(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            initial_latent=initial_latent
        )
        # Contiguous long rollout stashed by _run_generator (None unless use_memory_dmd);
        # used to build the AR teacher's history for the memory-effect term.
        student_full = getattr(self, "_last_rollout_full", None) if getattr(self, "use_memory_dmd", False) else None

        # Step 2: Compute the DMD loss
        dmd_loss, dmd_log_dict = self.compute_distribution_matching_loss(
            image_or_video=pred_image,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            gradient_mask=gradient_mask,
            denoised_timestep_from=denoised_timestep_from,
            denoised_timestep_to=denoised_timestep_to,
            student_full_latent=student_full,
        )

        return dmd_loss, dmd_log_dict

    def critic_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noise and train the critic with generated samples.
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
            - critic_log_dict: a dictionary containing the intermediate tensors for logging.
        """

        # Step 1: Run generator on backward simulated noisy input
        # The critic trains the fake score on the standard last-21 window; AR
        # refinement is only needed for the generator's real_score, so skip it here.
        with torch.no_grad():
            self._skip_segmentation = True
            try:
                generated_image, _, denoised_timestep_from, denoised_timestep_to = self._run_generator(
                    image_or_video_shape=image_or_video_shape,
                    conditional_dict=conditional_dict,
                    initial_latent=initial_latent
                )
            finally:
                self._skip_segmentation = False

        # Step 2: Compute the fake prediction
        min_timestep = denoised_timestep_to if self.ts_schedule and denoised_timestep_to is not None else self.min_score_timestep
        max_timestep = denoised_timestep_from if self.ts_schedule_max and denoised_timestep_from is not None else self.num_train_timestep
        critic_timestep = self._get_timestep(
            min_timestep,
            max_timestep,
            image_or_video_shape[0],
            image_or_video_shape[1],
            self.num_frame_per_block,
            uniform_timestep=True
        )

        if self.timestep_shift > 1:
            critic_timestep = self.timestep_shift * \
                (critic_timestep / 1000) / (1 + (self.timestep_shift - 1) * (critic_timestep / 1000)) * 1000

        critic_timestep = critic_timestep.clamp(self.min_step, self.max_step)

        critic_noise = torch.randn_like(generated_image)
        noisy_generated_image = self.scheduler.add_noise(
            generated_image.flatten(0, 1),
            critic_noise.flatten(0, 1),
            critic_timestep.flatten(0, 1)
        ).unflatten(0, image_or_video_shape[:2])

        _, pred_fake_image = self.fake_score(
            noisy_image_or_video=noisy_generated_image,
            conditional_dict=conditional_dict,
            timestep=critic_timestep
        )

        # Step 3: Compute the denoising loss for the fake critic
        if self.args.denoising_loss_type == "flow":
            from utils.wan_wrapper import WanDiffusionWrapper
            flow_pred = WanDiffusionWrapper._convert_x0_to_flow_pred(
                scheduler=self.scheduler,
                x0_pred=pred_fake_image.flatten(0, 1),
                xt=noisy_generated_image.flatten(0, 1),
                timestep=critic_timestep.flatten(0, 1)
            )
            pred_fake_noise = None
        else:
            flow_pred = None
            pred_fake_noise = self.scheduler.convert_x0_to_noise(
                x0=pred_fake_image.flatten(0, 1),
                xt=noisy_generated_image.flatten(0, 1),
                timestep=critic_timestep.flatten(0, 1)
            ).unflatten(0, image_or_video_shape[:2])

        denoising_loss = self.denoising_loss_func(
            x=generated_image.flatten(0, 1),
            x_pred=pred_fake_image.flatten(0, 1),
            noise=critic_noise.flatten(0, 1),
            noise_pred=pred_fake_noise,
            alphas_cumprod=self.scheduler.alphas_cumprod,
            timestep=critic_timestep.flatten(0, 1),
            flow_pred=flow_pred
        )

        # Step 5: Debugging Log
        critic_log_dict = {
            "critic_timestep": critic_timestep.detach()
        }

        return denoising_loss, critic_log_dict

    @torch.no_grad()
    def _prepare_generator_input(self, ode_latent: torch.Tensor, tf=False, causal = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given a tensor containing the whole ODE sampling trajectories,
        randomly choose an intermediate timestep and return the latent as well as the corresponding timestep.
        Input:
            - ode_latent: a tensor containing the whole ODE sampling trajectories [batch_size, num_denoising_steps, num_frames, num_channels, height, width].
        Output:
            - noisy_input: a tensor containing the selected latent [batch_size, num_frames, num_channels, height, width].
            - timestep: a tensor containing the corresponding timestep [batch_size].
        """
        batch_size, num_denoising_steps, num_frames, num_channels, height, width = ode_latent.shape

        # Step 1: Randomly choose a timestep for each frame
        uniform_timestep = tf or (not causal)
        index = self._get_timestep(
            0,
            len(self.denoising_step_list),
            batch_size,
            num_frames,
            self.num_frame_per_block,
            uniform_timestep=uniform_timestep
        )
        if self.args.i2v:
            index[:, 0] = len(self.denoising_step_list) - 1

        noisy_input = torch.gather(
            ode_latent, dim=1,
            index=index.reshape(batch_size, 1, num_frames, 1, 1, 1).expand(
                -1, -1, -1, num_channels, height, width).to(self.device)
        ).squeeze(1)

        timestep = self.denoising_step_list[index].to(self.device)

        return noisy_input, timestep