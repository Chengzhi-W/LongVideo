#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-python}"
NNODES="${NNODES:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29661}"

for value_name in NNODES NPROC_PER_NODE; do
  value="${!value_name}"
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || {
    echo "INVALID_POSITIVE_INTEGER:$value_name=$value" >&2
    exit 1
  }
done

case "$NNODES:$NPROC_PER_NODE" in
  1:8)
    AUTO_CONFIG="$BASE/configs/ar_teacher_ultravideo_long_5k_81f_sparse12_8gpu.yaml"
    NODE_RANK="${NODE_RANK:-0}"
    MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
    ;;
  2:8)
    AUTO_CONFIG="$BASE/configs/ar_teacher_ultravideo_long_5k_81f_sparse12.yaml"
    NODE_RANK="${NODE_RANK:?Set NODE_RANK=0 on the first machine and NODE_RANK=1 on the second}"
    MASTER_ADDR="${MASTER_ADDR:?Set MASTER_ADDR to the reachable IP of the first machine}"
    ;;
  *)
    echo "UNSUPPORTED_AR_TEACHER_TOPOLOGY:NNODES=$NNODES:NPROC_PER_NODE=$NPROC_PER_NODE" >&2
    echo "Supported topologies are one node x 8 GPUs and two nodes x 8 GPUs." >&2
    exit 1
    ;;
esac

[[ "$NODE_RANK" =~ ^[0-9]+$ ]] && (( NODE_RANK < NNODES )) || {
  echo "INVALID_NODE_RANK:NODE_RANK=$NODE_RANK:NNODES=$NNODES" >&2
  exit 1
}

WORLD_SIZE=$((NNODES * NPROC_PER_NODE))
CONFIG="${CONFIG:-$AUTO_CONFIG}"
LOGDIR="${LOGDIR:-$BASE/logs/ar_teacher_sparse12_20s_5k_${WORLD_SIZE}gpu_seed42}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONPATH="$BASE"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTHONUNBUFFERED=1
export WANDB_MODE="${WANDB_MODE:-online}"
export WAN_FLEX_COMPILE_MODE="${WAN_FLEX_COMPILE_MODE:-max-autotune-no-cudagraphs}"

[[ -f "$CONFIG" ]] || { echo "MISSING_REQUIRED_PATH:$CONFIG" >&2; exit 1; }

cd "$BASE"
"$PYTHON" tools/audit_sparse_teacher_forcing_20s.py --config "$CONFIG"
"$PYTHON" - "$CONFIG" "$NNODES" "$NPROC_PER_NODE" <<'PY'
import sys

from omegaconf import OmegaConf

config_path, nnodes, nproc_per_node = sys.argv[1:]
config = OmegaConf.load(config_path)
world_size = int(nnodes) * int(nproc_per_node)

expected = {
    8: {"sharding_strategy": "full", "gradient_accumulation_steps": 2},
    16: {"sharding_strategy": "hybrid_full", "gradient_accumulation_steps": 1},
}
if world_size not in expected:
    raise SystemExit(f"UNSUPPORTED_WORLD_SIZE:{world_size}")

contract = expected[world_size]
micro_batch = int(config.batch_size)
accumulation = int(getattr(config, "gradient_accumulation_steps", 1))
global_batch = world_size * micro_batch * accumulation
configured_global_batch = int(config.total_batch_size)
sharding = str(config.sharding_strategy)

if micro_batch != 1:
    raise SystemExit(f"MICRO_BATCH_MISMATCH:actual={micro_batch}:expected=1")
if accumulation != contract["gradient_accumulation_steps"]:
    raise SystemExit(
        "GRADIENT_ACCUMULATION_MISMATCH:"
        f"actual={accumulation}:expected={contract['gradient_accumulation_steps']}"
    )
if sharding != contract["sharding_strategy"]:
    raise SystemExit(
        f"SHARDING_STRATEGY_MISMATCH:actual={sharding}:"
        f"expected={contract['sharding_strategy']}"
    )
if global_batch != configured_global_batch or global_batch != 16:
    raise SystemExit(
        "WORLD_SIZE_GLOBAL_BATCH_MISMATCH:"
        f"actual={global_batch}:configured={configured_global_batch}:expected=16"
    )

print(
    "AR_TEACHER_DISTRIBUTED_AUDIT_OK:"
    f"world_size={world_size}:micro_batch={micro_batch}:"
    f"gradient_accumulation={accumulation}:global_batch={global_batch}:"
    f"sharding={sharding}:config={config_path}"
)
PY

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "AR_TEACHER_LAUNCH_DRY_RUN_OK:world_size=$WORLD_SIZE:node_rank=$NODE_RANK:config=$CONFIG"
  exit 0
fi

for path in \
  "$BASE/checkpoints/chunkwise/ar_diffusion.pt" \
  "$BASE/dataset/UltraVideo-Long-5K-20s/manifests/train.csv" \
  "$BASE/dataset/UltraVideo-Long-5K-20s/latents_81/_ENCODE_COMPLETE.json" \
  "$BASE/dataset/UltraVideo-Long-5K-20s/clean_data_81"; do
  [[ -e "$path" ]] || { echo "MISSING_REQUIRED_PATH:$path" >&2; exit 1; }
done

[[ ! -e "$LOGDIR/train_node${NODE_RANK}.log" ]] || {
  echo "REFUSE_TO_OVERWRITE_EXISTING_NODE_LOG:$LOGDIR/train_node${NODE_RANK}.log" >&2
  exit 1
}
mkdir -p "$LOGDIR/wandb"

"$PYTHON" tools/audit_ultravideo_long_20s_data.py \
  --manifest dataset/UltraVideo-Long-5K-20s/manifests/train.csv \
  --latents-dir dataset/UltraVideo-Long-5K-20s/latents_81 \
  --lmdb dataset/UltraVideo-Long-5K-20s/clean_data_81 \
  --expected-count 4900 \
  --latent-shape 81 16 60 104
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
