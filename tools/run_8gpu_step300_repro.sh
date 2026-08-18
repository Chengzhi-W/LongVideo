#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/Causal-Forcing
PYTHON=/root/autodl-tmp/envs/longlive/bin/python

cd "$BASE"
export PYTHONPATH="$BASE"
export WANDB_MODE=online
export OMP_NUM_THREADS=8

run_one() {
  local config_path="$1"
  local logdir="$2"
  local port="$3"

  mkdir -p "$logdir/wandb"
  "$PYTHON" -m torch.distributed.run \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port="$port" \
    train.py \
    --config_path "$config_path" \
    --logdir "$logdir" \
    --wandb-save-dir "$logdir/wandb" \
    2>&1 | tee "$logdir/train_run.log"

  test -s "$logdir/checkpoint_model_000300/model.pt"
}

run_one \
  configs/causal_forcing_dmd_chunkwise_senior_baseline.yaml \
  logsv1/repro8x_causal_forcing_dmd_seed_baseline_step300_chunkwise \
  29561

run_one \
  configs/causal_forcing_dmd_chunkwise_senior_mem1_nonorm.yaml \
  logsv1/repro8x_causal_forcing_dmd_seed_mem1_nonorm_step300_chunkwise \
  29562

echo ALL_8GPU_STEP300_TRAINING_COMPLETE
