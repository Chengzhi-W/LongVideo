#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-$BASE/configs/ar_teacher_ultravideo_long_5k_63f_2k.yaml}"
LOGDIR="${LOGDIR:-$BASE/logs/ar_teacher_ultravideo_long_5k_63f_2k_seed42}"
NNODES="${NNODES:-2}"
NODE_RANK="${NODE_RANK:?Set NODE_RANK=0 on the first machine and NODE_RANK=1 on the second}"
MASTER_ADDR="${MASTER_ADDR:?Set MASTER_ADDR to the reachable IP of the first machine}"
MASTER_PORT="${MASTER_PORT:-29641}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONPATH="$BASE"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTHONUNBUFFERED=1
export WANDB_MODE="${WANDB_MODE:-online}"
# The 1.3B model's 128-dim FlexAttention heads need autotuning on some GPU
# generations (notably Blackwell) to avoid an invalid shared-memory schedule.
export WAN_FLEX_COMPILE_MODE="${WAN_FLEX_COMPILE_MODE:-max-autotune-no-cudagraphs}"

for path in \
  "$CONFIG" \
  "$BASE/checkpoints/chunkwise/ar_diffusion.pt" \
  "$BASE/dataset/UltraVideo-Long-5K/clean_data_63"; do
  [[ -e "$path" ]] || { echo "MISSING_REQUIRED_PATH:$path" >&2; exit 1; }
done

[[ ! -e "$LOGDIR/train_node${NODE_RANK}.log" ]] || {
  echo "REFUSE_TO_OVERWRITE_EXISTING_NODE_LOG:$LOGDIR/train_node${NODE_RANK}.log" >&2
  exit 1
}
mkdir -p "$LOGDIR/wandb"

cd "$BASE"
"$PYTHON" tools/audit_sink_recent_teacher_forcing.py --config "$CONFIG"
"$PYTHON" -m torch.distributed.run \
  --nproc_per_node="$NPROC_PER_NODE" \
  --nnodes="$NNODES" \
  --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  train.py \
  --config_path "$CONFIG" \
  --logdir "$LOGDIR" \
  --wandb-save-dir "$LOGDIR/wandb" \
  --no_visualize \
  2>&1 | tee "$LOGDIR/train_node${NODE_RANK}.log"
