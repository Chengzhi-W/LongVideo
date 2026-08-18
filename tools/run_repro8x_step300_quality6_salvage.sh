#!/usr/bin/env bash
set -euo pipefail

BASE=${BASE:-/root/autodl-tmp/Causal-Forcing}
VBENCH_ROOT=${VBENCH_ROOT:-/root/autodl-tmp/VBench}
GEN_PY=${GEN_PY:-/root/autodl-tmp/envs/longlive/bin/python}
EVAL_PY=${EVAL_PY:-/root/autodl-tmp/envs/vbench/bin/python}
MANIFEST="$BASE/prompts/vbench_quality_no_tf_seed42.json"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

DIMS=(
  subject_consistency
  background_consistency
  motion_smoothness
  aesthetic_quality
  imaging_quality
  dynamic_degree
)

video_count() {
  find "$1" -maxdepth 1 -type f -name '*.mp4' -size +0c | wc -l
}

hardlink_videos() {
  local source="$1"
  local target="$2"
  [[ -d "$source" ]] || return 0
  find "$source" -maxdepth 1 -type f -name '*.mp4' -size +0c -print0 |
    while IFS= read -r -d '' video; do
      local destination="$target/$(basename "$video")"
      [[ -e "$destination" ]] || ln "$video" "$destination"
    done
}

generate_suite() {
  local kind="$1"
  local config="$2"
  local checkpoint="$3"
  local videos="$4"
  local suite="$5"
  local expected_total="$6"

  cd "$BASE"
  export PYTHONPATH="$BASE:$BASE/long_video"
  echo "START_GENERATION:$kind:$suite:step0300:8gpu:batch4:base_seed0:63_latent"
  "$GEN_PY" -m torch.distributed.run --standalone --nproc_per_node=8 \
    tools/vbench_generate.py \
    --config_path "$config" \
    --checkpoint_path "$checkpoint" \
    --manifest "$MANIFEST" \
    --output_folder "$videos" \
    --suite "$suite" \
    --num_output_frames 63 \
    --samples_per_prompt 5 \
    --batch_size 4 \
    --base_seed 0 \
    --seed_mode prompt_batch_stream \
    --rank_seed_offset \
    --use_ema

  local count
  count=$(video_count "$videos")
  [[ "$count" -eq "$expected_total" ]] || {
    echo "EXPECTED_${expected_total}_VIDEOS_AFTER_${suite}:found${count}" >&2
    exit 1
  }
  echo "GENERATION_COMPLETE:$kind:$suite:$count"
}

evaluate_dimension() {
  local kind="$1"
  local videos="$2"
  local out="$3"
  local dim="$4"
  local dim_out="$out/by_dimension/$dim"
  mkdir -p "$dim_out"

  local existing
  existing=$(find "$dim_out" -maxdepth 1 -type f -name '*_eval_results.json' -print -quit)
  if [[ -n "$existing" ]]; then
    echo "SKIP_COMPLETE:$kind:$dim:$existing"
    return
  fi

  cd "$VBENCH_ROOT"
  export PYTHONPATH="$VBENCH_ROOT"
  echo "START_DIMENSION:$kind:$dim:8gpu"
  "$EVAL_PY" -m torch.distributed.run --standalone --nproc_per_node=8 evaluate.py \
    --videos_path "$videos" \
    --output_path "$dim_out" \
    --dimension "$dim" \
    --load_ckpt_from_local True \
    --mode vbench_standard
  local result
  result=$(find "$dim_out" -maxdepth 1 -type f -name '*_eval_results.json' -print -quit)
  [[ -n "$result" ]] || { echo "MISSING_RESULT:$kind:$dim" >&2; exit 1; }
  echo "DONE_DIMENSION:$kind:$dim:$result"
}

prepare_model() {
  local kind="$1"
  local config="$2"
  local checkpoint="$3"
  local subject_source="$4"
  local overall_source="$5"
  local videos="$6"

  [[ -s "$checkpoint" ]] || { echo "MISSING_CHECKPOINT:$checkpoint" >&2; exit 1; }
  mkdir -p "$videos"

  hardlink_videos "$subject_source" "$videos"
  hardlink_videos "$overall_source" "$videos"
  echo "REUSED_VIDEOS:$kind:$(video_count "$videos")"

  local count
  count=$(video_count "$videos")
  if [[ "$count" -lt 825 ]]; then
    generate_suite "$kind" "$config" "$checkpoint" "$videos" overall_consistency 825
  fi
  count=$(video_count "$videos")
  if [[ "$count" -lt 1255 ]]; then
    generate_suite "$kind" "$config" "$checkpoint" "$videos" scene 1255
  fi
  [[ "$(video_count "$videos")" -eq 1255 ]] || { echo "EXPECTED_1255:$kind" >&2; exit 1; }
  echo "ALL_VIDEOS_READY:$kind:1255"
}

evaluate_model() {
  local kind="$1"
  local videos="$2"
  local out="$3"

  [[ "$(video_count "$videos")" -eq 1255 ]] || { echo "EVAL_REQUIRES_1255:$kind" >&2; exit 1; }
  mkdir -p "$out/by_dimension"

  for dim in "${DIMS[@]}"; do
    evaluate_dimension "$kind" "$videos" "$out" "$dim"
  done

  mapfile -t result_files < <(find "$out/by_dimension" -mindepth 2 -maxdepth 2 \
    -type f -name '*_eval_results.json' | sort)
  [[ "${#result_files[@]}" -eq 6 ]] || {
    echo "EXPECTED_6_RESULTS:$kind:found${#result_files[@]}" >&2
    exit 1
  }
  "$EVAL_PY" "$BASE/tools/merge_vbench_dimension_results.py" \
    --output "$out/combined_eval_results.json" "${result_files[@]}"
  "$EVAL_PY" "$BASE/tools/calc_vbench_quality_no_tf.py" \
    "$out/combined_eval_results.json" > "$out/quality_summary_no_tf.json"
  touch "$out/COMPLETE"
  echo "QUALITY6_COMPLETE:$kind:$out/quality_summary_no_tf.json"
}

# Phase 1: generate every missing video for both models before any evaluation.
prepare_model \
  mem1_nonorm \
  "$BASE/configs/causal_forcing_dmd_chunkwise_senior_mem1_nonorm.yaml" \
  "$BASE/logsv1/repro8x_causal_forcing_dmd_seed_mem1_nonorm_step300_chunkwise/checkpoint_model_000300/model.pt" \
  "$BASE/outputsv1/vbench_repro8x_mem1_step0300_15s_seed0_subject" \
  "$BASE/outputsv1/vbench_repro8x_mem1_step0300_15s_seed0_quality_rest5" \
  "$BASE/outputsv1/vbench_repro8x_mem1_step0300_15s_seed0_quality6_reused"

prepare_model \
  baseline \
  "$BASE/configs/causal_forcing_dmd_chunkwise_senior_baseline.yaml" \
  "$BASE/logsv1/repro8x_causal_forcing_dmd_seed_baseline_step300_chunkwise/checkpoint_model_000300/model.pt" \
  "$BASE/outputsv1/vbench_repro8x_baseline_step0300_15s_seed0_subject" \
  "$BASE/outputsv1/vbench_repro8x_baseline_step0300_15s_seed0_quality_rest5" \
  "$BASE/outputsv1/vbench_repro8x_baseline_step0300_15s_seed0_quality6_reused"

echo ALL_VIDEO_GENERATION_COMPLETE:mem1_nonorm=1255:baseline=1255

# Phase 2: only after both video sets are complete, evaluate all six dimensions.
evaluate_model \
  mem1_nonorm \
  "$BASE/outputsv1/vbench_repro8x_mem1_step0300_15s_seed0_quality6_reused" \
  "$BASE/outputsv1/vbench_results_repro8x_mem1_step0300_15s_seed0_quality6_reused_no_tf"

evaluate_model \
  baseline \
  "$BASE/outputsv1/vbench_repro8x_baseline_step0300_15s_seed0_quality6_reused" \
  "$BASE/outputsv1/vbench_results_repro8x_baseline_step0300_15s_seed0_quality6_reused_no_tf"

echo ALL_REPRO8X_STEP300_QUALITY6_REUSED_COMPLETE
