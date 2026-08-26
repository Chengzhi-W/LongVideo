#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-$BASE/configs/ar_teacher_ultravideo_long_5k_81f_sparse12.yaml}"
LOGDIR="${LOGDIR:-$BASE/logs/ar_teacher_sparse12_20s_5k_seed42}"
NNODES="${NNODES:-2}"
NODE_RANK="${NODE_RANK:?Set NODE_RANK=0 on the first machine and NODE_RANK=1 on the second}"
MASTER_ADDR="${MASTER_ADDR:?Set MASTER_ADDR to the reachable IP of the first machine}"
MASTER_PORT="${MASTER_PORT:-29661}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONPATH="$BASE"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTHONUNBUFFERED=1
export WANDB_MODE="${WANDB_MODE:-online}"
export WAN_FLEX_COMPILE_MODE="${WAN_FLEX_COMPILE_MODE:-max-autotune-no-cudagraphs}"

for path in \
  "$CONFIG" \
  "$BASE/checkpoints/chunkwise/ar_diffusion.pt" \
  "$BASE/dataset/UltraVideo-Long-5K/clean_data_81"; do
  [[ -e "$path" ]] || { echo "MISSING_REQUIRED_PATH:$path" >&2; exit 1; }
done

[[ ! -e "$LOGDIR/train_node${NODE_RANK}.log" ]] || {
  echo "REFUSE_TO_OVERWRITE_EXISTING_NODE_LOG:$LOGDIR/train_node${NODE_RANK}.log" >&2
  exit 1
}
mkdir -p "$LOGDIR/wandb"

cd "$BASE"
"$PYTHON" tools/audit_sparse_teacher_forcing_20s.py --config "$CONFIG"
"$PYTHON" - "$CONFIG" "$NNODES" "$NPROC_PER_NODE" <<'PY'
import sys
from omegaconf import OmegaConf

config = OmegaConf.load(sys.argv[1])
world_size = int(sys.argv[2]) * int(sys.argv[3])
effective_global_batch = world_size * int(config.batch_size)
if effective_global_batch != int(config.total_batch_size):
    raise SystemExit(
        "WORLD_SIZE_GLOBAL_BATCH_MISMATCH: "
        f"actual={effective_global_batch}, configured={config.total_batch_size}; "
        "the diffusion trainer has no gradient accumulation"
    )
print(
    "AR_TEACHER_DISTRIBUTED_AUDIT_OK:"
    f"world_size={world_size}:global_batch={effective_global_batch}"
)
PY
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
