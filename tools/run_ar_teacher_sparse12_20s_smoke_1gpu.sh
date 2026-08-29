#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-$BASE/configs/ar_teacher_ultravideo_long_smoke_81f_sparse12_1gpu.yaml}"
LOGDIR="${LOGDIR:-$BASE/logs/ar_teacher_sparse12_20s_smoke_seed42}"
PORT="${MASTER_PORT:-29651}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$BASE"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
export WAN_FLEX_COMPILE_MODE="${WAN_FLEX_COMPILE_MODE:-max-autotune-no-cudagraphs}"

for path in \
  "$CONFIG" \
  "$BASE/checkpoints/chunkwise/ar_diffusion.pt" \
  "$BASE/dataset/UltraVideo-Long-debug/latents_81_240x416/_ENCODE_COMPLETE.json" \
  "$BASE/dataset/UltraVideo-Long-debug/clean_data_81_240x416"; do
  [[ -e "$path" ]] || { echo "MISSING_REQUIRED_PATH:$path" >&2; exit 1; }
done

[[ ! -e "$LOGDIR" ]] || {
  echo "REFUSE_TO_OVERWRITE_EXISTING_SMOKE:$LOGDIR" >&2
  exit 1
}
mkdir -p "$LOGDIR"

cd "$BASE"
"$PYTHON" tools/audit_sparse_teacher_forcing_20s.py --config "$CONFIG"
"$PYTHON" tools/audit_ultravideo_long_20s_data.py \
  --manifest dataset/UltraVideo-Long-debug/manifests/train.csv \
  --videos-dir dataset/UltraVideo-Long-debug/videos \
  --latents-dir dataset/UltraVideo-Long-debug/latents_81_240x416 \
  --lmdb dataset/UltraVideo-Long-debug/clean_data_81_240x416 \
  --expected-count 4 \
  --latent-shape 81 16 30 52
"$PYTHON" -m torch.distributed.run \
  --nproc_per_node=1 \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --master_port="$PORT" \
  train.py \
  --config_path "$CONFIG" \
  --logdir "$LOGDIR" \
  --wandb-save-dir "$LOGDIR/wandb" \
  --disable-wandb \
  --no_visualize \
  2>&1 | tee "$LOGDIR/train.log"

for step in 000001 000002; do
  [[ -s "$LOGDIR/checkpoint_model_${step}/model.pt" ]] || {
    echo "MISSING_SMOKE_CHECKPOINT:step=$step" >&2
    exit 1
  }
done
grep -q 'Reached max_steps=2' "$LOGDIR/train.log" || {
  echo "SMOKE_DID_NOT_REACH_MAX_STEPS" >&2
  exit 1
}
"$PYTHON" tools/audit_ar_teacher_checkpoint.py \
  --base "$BASE/checkpoints/chunkwise/ar_diffusion.pt" \
  --tuned "$LOGDIR/checkpoint_model_000002/model.pt"
echo "AR_TEACHER_SPARSE12_20S_SMOKE_COMPLETE:$LOGDIR"
