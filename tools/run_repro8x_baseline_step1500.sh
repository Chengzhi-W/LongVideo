#!/usr/bin/env bash
set -euo pipefail

BASE=${BASE:-/root/autodl-tmp/Causal-Forcing}
PYTHON=${PYTHON:-/root/autodl-tmp/envs/longlive/bin/python}
CONFIG="$BASE/configs/causal_forcing_dmd_chunkwise_senior_baseline_step1500.yaml"
LOGDIR="$BASE/logsv1/repro8x_causal_forcing_dmd_seed42_baseline_step1500_chunkwise"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH="$BASE"
export WANDB_MODE=online
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

[[ -s "$CONFIG" ]] || { echo "MISSING_CONFIG:$CONFIG" >&2; exit 1; }
[[ ! -e "$LOGDIR/train_run.log" ]] || {
  echo "REFUSE_TO_OVERWRITE_EXISTING_RUN:$LOGDIR" >&2
  exit 1
}

"$PYTHON" - "$CONFIG" <<'PY'
import sys
from omegaconf import OmegaConf

c = OmegaConf.load(sys.argv[1])
assert c.seed == 42, c.seed
assert c.max_steps == 1500, c.max_steps
assert list(c.save_steps) == [300, 600, 900, 1200, 1500], c.save_steps
assert c.use_memory_dmd is False, c.use_memory_dmd
assert c.batch_size == 1, c.batch_size
assert c.total_batch_size == 64, c.total_batch_size
assert c.num_training_frames == 63, c.num_training_frames
assert c.ema_weight == 0.99, c.ema_weight
print("CONFIG_AUDIT:baseline:seed42:max1500:saves300,600,900,1200,1500:8gpu:batch1:total64:EMA:memory_false")
PY

mkdir -p "$LOGDIR/wandb"
cd "$BASE"

"$PYTHON" -m torch.distributed.run \
  --nproc_per_node=8 \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --master_port=29571 \
  train.py \
  --config_path "$CONFIG" \
  --logdir "$LOGDIR" \
  --wandb-save-dir "$LOGDIR/wandb" \
  2>&1 | tee "$LOGDIR/train_run.log"

for step in 000300 000600 000900 001200 001500; do
  [[ -s "$LOGDIR/checkpoint_model_${step}/model.pt" ]] || {
    echo "MISSING_CHECKPOINT:$step" >&2
    exit 1
  }
done

count=$(find "$LOGDIR" -mindepth 2 -maxdepth 2 -type f \
  -path '*/checkpoint_model_*/model.pt' -size +0c | wc -l)
[[ "$count" -eq 5 ]] || { echo "EXPECTED_EXACTLY_5_CHECKPOINTS:found${count}" >&2; exit 1; }
echo "BASELINE_STEP1500_COMPLETE:5_CHECKPOINTS"
