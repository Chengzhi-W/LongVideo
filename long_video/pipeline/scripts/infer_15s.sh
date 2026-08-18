#!/bin/bash
# 15s (63 latent frame) inference for the chunkwise causal-forcing DMD model.
# 63 = 21 (5s) * 3, divisible by num_frame_per_block=3. Output @ 16fps -> ~15s.
# Uses the tiny step-20 checkpoint (generator key only -> no --use_ema) purely to
# verify the 15s rollout path runs end-to-end; quality is not expected to be good.

cd /apdcephfs_gy31/share_301869871/taiji/xxuanhe/project/Causal-Forcing
export PYTHONPATH='/apdcephfs_gy31/share_301869871/taiji/xxuanhe/project/Causal-Forcing'

CKPT=${CKPT:-logsv1/causal_forcing_dmd_seed_chunkwise/checkpoint_model_000900/model.pt}
OUT=${OUT:-outputsv1/infer_30s_dmd_seed_step900}
NUM_FRAMES=${NUM_FRAMES:-126}   # 63 latent frames ≈ 15s @ 16fps

export CUDA_VISIBLE_DEVICES=2
python inference.py \
    --config_path configs/infer_15s_chunkwise.yaml \
    --checkpoint_path "${CKPT}" \
    --data_path prompts/demos.txt \
    --output_folder "${OUT}" \
    --num_output_frames "${NUM_FRAMES}" \
    --seed 0 \
    --use_ema
