#!/bin/bash
# Inference the chunk-wise AR-diffusion TEACHER (ar_diffusion.pt) over a long
# rollout, to inspect its temporal continuity / drift on its own.
#
# ar_diffusion.pt is a MULTI-STEP AR diffusion model (Stage-1 TF causalisation),
# so its ckpt has key `generator` (NOT generator_ema) and its config has NO
# denoising_step_list -> inference.py routes it to CausalDiffusionInferencePipeline
# (per-chunk 50-step UniPC + CFG). Do NOT pass --use_ema (there is no EMA here).
#
# NUM_FRAMES=63 latent frames = 21 (5s) * 3 -> ~15s @ 16fps, divisible by
# num_frame_per_block=3.

cd /apdcephfs_gy31/share_301869871/taiji/xxuanhe/project/Causal-Forcing
export PYTHONPATH='/apdcephfs_gy31/share_301869871/taiji/xxuanhe/project/Causal-Forcing'

CKPT=${CKPT:-/apdcephfs_gy31/share_301869871/taiji/xxuanhe/models/Causal-Forcing/chunkwise/ar_diffusion.pt}
OUT=${OUT:-outputsv1/infer_ar_teacher_15s_12steps}
NUM_FRAMES=${NUM_FRAMES:-63}   # 63 latent frames ≈ 15s @ 16fps

python inference.py \
    --config_path configs/infer_ar_teacher_chunkwise.yaml \
    --checkpoint_path "${CKPT}" \
    --data_path prompts/demos_3.txt \
    --output_folder "${OUT}" \
    --num_output_frames "${NUM_FRAMES}" \
    --seed 0
