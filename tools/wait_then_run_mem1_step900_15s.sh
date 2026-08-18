#!/usr/bin/env bash
set -euo pipefail

BASE=${BASE:-/root/autodl-tmp/Causal-Forcing}
BASELINE_COMPLETE="$BASE/outputsv1/vbench_results_baseline_step0900_15s_seed0_no_tf/COMPLETE"

echo "WAITING_FOR_BASELINE_STEP0900_15S_COMPLETE"
while [[ ! -f "$BASELINE_COMPLETE" ]]; do
  sleep 30
done

echo "BASELINE_COMPLETE_STARTING_MEM1_NONORM_STEP0900_15S"
exec bash "$BASE/tools/run_vbench_mem1_step900_15s_all6.sh"
