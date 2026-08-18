#!/usr/bin/env bash
set -euo pipefail

BASE=${BASE:-/root/autodl-tmp/Causal-Forcing}
VBENCH_ROOT=${VBENCH_ROOT:-/root/autodl-tmp/VBench}
GEN_PY=${GEN_PY:-/root/autodl-tmp/envs/longlive/bin/python}
EVAL_PY=${EVAL_PY:-/root/autodl-tmp/envs/vbench/bin/python}
MANIFEST="$BASE/prompts/vbench_quality_no_tf_seed42.json"
CONFIG="$BASE/configs/causal_forcing_dmd_chunkwise_senior_baseline.yaml"
CHECKPOINT="$BASE/logsv1/repro8x_causal_forcing_dmd_seed_baseline_step300_chunkwise/checkpoint_model_000300/model.pt"
VIDEOS="$BASE/outputsv1/vbench_repro8x_baseline_step0300_15s_allrankseed0_subject"
OUT="$BASE/outputsv1/vbench_results_repro8x_baseline_step0300_15s_allrankseed0_subject3"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

video_count() {
  find "$VIDEOS" -maxdepth 1 -type f -name '*.mp4' -size +0c | wc -l
}

[[ -s "$CHECKPOINT" ]] || { echo "MISSING_CHECKPOINT:$CHECKPOINT" >&2; exit 1; }
mkdir -p "$VIDEOS" "$OUT/by_dimension"

cd "$BASE"
export PYTHONPATH="$BASE:$BASE/long_video"
echo "PROTOCOL:baseline:step0300:EMA:8gpu:batch4:all_rank_seed0:63_latent:249_pixel_frames"
echo "START_GENERATION:baseline:subject_consistency:360"
"$GEN_PY" -m torch.distributed.run --standalone --nproc_per_node=8 \
  tools/vbench_generate.py \
  --config_path "$CONFIG" \
  --checkpoint_path "$CHECKPOINT" \
  --manifest "$MANIFEST" \
  --output_folder "$VIDEOS" \
  --suite subject_consistency \
  --num_output_frames 63 \
  --samples_per_prompt 5 \
  --batch_size 4 \
  --base_seed 0 \
  --seed_mode prompt_batch_stream \
  --use_ema

count=$(video_count)
[[ "$count" -eq 360 ]] || { echo "EXPECTED_360_SUBJECT_VIDEOS:found${count}" >&2; exit 1; }
echo "GENERATION_COMPLETE:baseline:$count"

evaluate_dimension() {
  local dim="$1"
  local dim_out="$OUT/by_dimension/$dim"
  mkdir -p "$dim_out"
  local existing
  existing=$(find "$dim_out" -maxdepth 1 -type f -name '*_eval_results.json' -print -quit)
  if [[ -n "$existing" ]]; then
    echo "SKIP_COMPLETE:baseline:$dim:$existing"
    return
  fi

  cd "$VBENCH_ROOT"
  export PYTHONPATH="$VBENCH_ROOT"
  echo "START_DIMENSION:baseline:$dim:8gpu"
  "$EVAL_PY" -m torch.distributed.run --standalone --nproc_per_node=8 evaluate.py \
    --videos_path "$VIDEOS" \
    --output_path "$dim_out" \
    --dimension "$dim" \
    --load_ckpt_from_local True \
    --mode vbench_standard
  local result
  result=$(find "$dim_out" -maxdepth 1 -type f -name '*_eval_results.json' -print -quit)
  [[ -n "$result" ]] || { echo "MISSING_RESULT:baseline:$dim" >&2; exit 1; }
  echo "DONE_DIMENSION:baseline:$dim:$result"
}

evaluate_dimension subject_consistency
evaluate_dimension motion_smoothness
evaluate_dimension dynamic_degree

mapfile -t results < <(find "$OUT/by_dimension" -mindepth 2 -maxdepth 2 \
  -type f -name '*_eval_results.json' | sort)
[[ "${#results[@]}" -eq 3 ]] || { echo "EXPECTED_3_RESULTS:found${#results[@]}" >&2; exit 1; }
"$EVAL_PY" "$BASE/tools/merge_vbench_dimension_results.py" \
  --output "$OUT/combined_eval_results.json" "${results[@]}"
touch "$OUT/COMPLETE"
echo "ALL_BASELINE_STEP300_ALLRANKSEED0_SUBJECT3_COMPLETE"
