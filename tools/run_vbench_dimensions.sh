#!/usr/bin/env bash
set -euo pipefail

BASE=${BASE:-/root/autodl-tmp/Causal-Forcing}
VBENCH_ROOT=${VBENCH_ROOT:-/root/autodl-tmp/VBench}
PYTHON=${PYTHON:-/root/autodl-tmp/envs/vbench/bin/python}
STEP=${STEP:-0900}
VIDEOS="$BASE/outputsv1/vbench_mem1_step${STEP}"
OUT="$BASE/outputsv1/vbench_results_mem1_step${STEP}_no_tf"
DIMS=(
  subject_consistency
  background_consistency
  motion_smoothness
  aesthetic_quality
  imaging_quality
  dynamic_degree
)

mkdir -p "$OUT/by_dimension"
cd "$VBENCH_ROOT"
export PYTHONPATH="$VBENCH_ROOT"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

for dim in "${DIMS[@]}"; do
  dim_out="$OUT/by_dimension/$dim"
  mkdir -p "$dim_out"
  existing=$(find "$dim_out" -maxdepth 1 -type f -name '*_eval_results.json' -print -quit)
  if [[ -n "$existing" ]]; then
    echo "SKIP_COMPLETE:$dim:$existing"
    continue
  fi
  echo "START_DIMENSION:$dim"
  "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 evaluate.py \
    --videos_path "$VIDEOS" \
    --output_path "$dim_out" \
    --dimension "$dim" \
    --load_ckpt_from_local True \
    --mode vbench_standard
  result=$(find "$dim_out" -maxdepth 1 -type f -name '*_eval_results.json' -print -quit)
  [[ -n "$result" ]] || { echo "MISSING_RESULT:$dim"; exit 1; }
  echo "DONE_DIMENSION:$dim:$result"
done

mapfile -t result_files < <(find "$OUT/by_dimension" -mindepth 2 -maxdepth 2 -type f -name '*_eval_results.json' | sort)
"$PYTHON" "$BASE/tools/merge_vbench_dimension_results.py" \
  --output "$OUT/combined_eval_results.json" "${result_files[@]}"
"$PYTHON" "$BASE/tools/calc_vbench_quality_no_tf.py" \
  "$OUT/combined_eval_results.json" > "$OUT/quality_summary_no_tf.json"
touch "$OUT/STEP${STEP}_EVAL_COMPLETE"
echo "ALL_DIMENSIONS_COMPLETE:$STEP"
