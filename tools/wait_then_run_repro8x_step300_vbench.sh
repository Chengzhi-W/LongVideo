#!/usr/bin/env bash
set -euo pipefail

BASE=${BASE:-/root/autodl-tmp/Causal-Forcing}
BASELINE_CKPT="$BASE/logsv1/repro8x_causal_forcing_dmd_seed_baseline_step300_chunkwise/checkpoint_model_000300/model.pt"
MEM1_CKPT="$BASE/logsv1/repro8x_causal_forcing_dmd_seed_mem1_nonorm_step300_chunkwise/checkpoint_model_000300/model.pt"

cd "$BASE"
while [[ ! -s "$BASELINE_CKPT" || ! -s "$MEM1_CKPT" ]]; do
  sleep 60
done

# A checkpoint appears during the final save. Wait for all training workers to
# finish so evaluation never contends with an active 8-GPU training process.
while pgrep -af 'train.py --config_path configs/causal_forcing_dmd_chunkwise_senior_(baseline|mem1_nonorm).yaml' >/dev/null; do
  sleep 30
done

mkdir -p outputsv1
bash tools/run_vbench_repro8x_step300_15s_all6.sh baseline 0,1,2,3 \
  > outputsv1/vbench_repro8x_baseline_step0300_15s_all6.log 2>&1 &
baseline_pid=$!
bash tools/run_vbench_repro8x_step300_15s_all6.sh mem1_nonorm 4,5,6,7 \
  > outputsv1/vbench_repro8x_mem1_step0300_15s_all6.log 2>&1 &
mem1_pid=$!

status=0
wait "$baseline_pid" || status=1
wait "$mem1_pid" || status=1
if [[ "$status" -ne 0 ]]; then
  echo VBench_failed >&2
  exit "$status"
fi
echo ALL_REPRO8X_STEP300_VBENCH_COMPLETE
