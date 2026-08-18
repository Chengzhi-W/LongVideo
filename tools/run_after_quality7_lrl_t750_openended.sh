#!/usr/bin/env bash
set -euo pipefail

BASE=${BASE:-/root/autodl-tmp/Causal-Forcing}
PYTHON=${PYTHON:-/root/autodl-tmp/envs/longlive/bin/python}
CONFIG="$BASE/configs/causal_forcing_dmd_chunkwise_long_vs_local_joint_t750_openended.yaml"
LOGDIR="$BASE/logsv1/repro8x_long_vs_local_joint_t750_openended_seed42"
METHOD_DONE="$BASE/outputsv1/vbench_lrl_nogate_step0600_15s_allrankseed0_quality7_results/COMPLETE"
BASELINE_DONE="$BASE/outputsv1/vbench_baseline_step0600_15s_allrankseed0_quality7_results/COMPLETE"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH="$BASE"
export WANDB_MODE=online
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

echo "WAIT_STEP600_QUALITY7_PAIR"
until [[ -f "$METHOD_DONE" && -f "$BASELINE_DONE" ]]; do
  sleep 30
done
echo "STEP600_QUALITY7_PAIR_CONFIRMED"

for path in "$CONFIG" "$BASE/checkpoints/chunkwise/causal_ode.pt" \
  "$BASE/checkpoints/chunkwise/ar_diffusion.pt"; do
  [[ -s "$path" ]] || { echo "MISSING_REQUIRED_FILE:$path" >&2; exit 1; }
done

"$PYTHON" - "$CONFIG" <<'PY'
import sys
from omegaconf import OmegaConf

c = OmegaConf.load(sys.argv[1])
assert c.seed == 42, c.seed
assert int(c.max_steps) == 0, c.max_steps
assert c.use_memory_dmd is True, c.use_memory_dmd
assert c.memory_control_type == "local_history", c.memory_control_type
assert c.memory_full_frames == 18, c.memory_full_frames
assert c.memory_local_frames == 6, c.memory_local_frames
assert c.memory_dmd_scale == 1.0, c.memory_dmd_scale
assert c.memory_max_timestep == 750, c.memory_max_timestep
assert c.memory_sink_frames == 0, c.memory_sink_frames
assert c.memory_highpass is False, c.memory_highpass
assert c.memory_joint_normalization is True, c.memory_joint_normalization
assert c.batch_size == 1, c.batch_size
assert c.total_batch_size == 64, c.total_batch_size
assert c.num_training_frames == 63, c.num_training_frames
assert c.ema_weight == 0.99, c.ema_weight
saves = [int(x) for x in c.save_steps]
assert saves and saves[0] == 300, saves[:3]
assert all(b - a == 300 for a, b in zip(saves, saves[1:])), saves[:10]
assert saves[-1] >= 99000, saves[-1]
print("CONFIG_AUDIT:A18-A6:t<750:joint_norm:scale1:seed42:open_ended:save_every300:8gpu:batch1:total64:EMA")
PY

[[ ! -e "$LOGDIR/train_run.log" ]] || {
  echo "REFUSE_TO_OVERWRITE_EXISTING_RUN:$LOGDIR" >&2
  exit 1
}

mkdir -p "$LOGDIR/wandb"
cd "$BASE"
echo "START_LRL_T750_OPENENDED_TRAINING:$LOGDIR"

"$PYTHON" -m torch.distributed.run \
  --nproc_per_node=8 \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --master_port=29593 \
  train.py \
  --config_path "$CONFIG" \
  --logdir "$LOGDIR" \
  --wandb-save-dir "$LOGDIR/wandb" \
  2>&1 | tee "$LOGDIR/train_run.log"
