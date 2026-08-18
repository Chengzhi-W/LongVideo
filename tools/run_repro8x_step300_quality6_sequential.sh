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

run_model() {
  local kind="$1"
  local config="$2"
  local checkpoint="$3"
  local videos="$4"
  local out="$5"

  [[ -s "$checkpoint" ]] || { echo "MISSING_CHECKPOINT:$checkpoint" >&2; exit 1; }
  mkdir -p "$videos" "$out/by_dimension"

  cd "$BASE"
  export PYTHONPATH="$BASE:$BASE/long_video"
  echo "START_OFFICIAL_ALL6_GENERATION:$kind:step0300:8gpu:batch5:base_seed0:63_latent"
  "$GEN_PY" -m torch.distributed.run --standalone --nproc_per_node=8 \
    tools/vbench_generate.py \
    --config_path "$config" \
    --checkpoint_path "$checkpoint" \
    --manifest "$MANIFEST" \
    --output_folder "$videos" \
    --num_output_frames 63 \
    --samples_per_prompt 5 \
    --batch_size 5 \
    --base_seed 0 \
    --seed_mode prompt_batch_stream \
    --rank_seed_offset \
    --use_ema

  local count
  count=$(find "$videos" -maxdepth 1 -type f -name '*.mp4' -size +0c | wc -l)
  [[ "$count" -eq 1255 ]] || { echo "EXPECTED_1255_OFFICIAL_VIDEOS:found${count}" >&2; exit 1; }
  echo "OFFICIAL_ALL6_GENERATION_COMPLETE:$kind:1255"

  cd "$VBENCH_ROOT"
  export PYTHONPATH="$VBENCH_ROOT"
  for dim in "${DIMS[@]}"; do
    local dim_out="$out/by_dimension/$dim"
    mkdir -p "$dim_out"
    local existing
    existing=$(find "$dim_out" -maxdepth 1 -type f -name '*_eval_results.json' -print -quit)
    if [[ -n "$existing" ]]; then
      echo "SKIP_COMPLETE:$kind:$dim:$existing"
      continue
    fi
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

# Strictly sequential: mem1_nonorm owns all 8 GPUs first, then baseline owns all 8.
run_model \
  mem1_nonorm \
  "$BASE/configs/causal_forcing_dmd_chunkwise_senior_mem1_nonorm.yaml" \
  "$BASE/logsv1/repro8x_causal_forcing_dmd_seed_mem1_nonorm_step300_chunkwise/checkpoint_model_000300/model.pt" \
  "$BASE/outputsv1/vbench_repro8x_mem1_step0300_15s_seed0_official_all6" \
  "$BASE/outputsv1/vbench_results_repro8x_mem1_step0300_15s_seed0_official_all6_no_tf"

run_model \
  baseline \
  "$BASE/configs/causal_forcing_dmd_chunkwise_senior_baseline.yaml" \
  "$BASE/logsv1/repro8x_causal_forcing_dmd_seed_baseline_step300_chunkwise/checkpoint_model_000300/model.pt" \
  "$BASE/outputsv1/vbench_repro8x_baseline_step0300_15s_seed0_official_all6" \
  "$BASE/outputsv1/vbench_results_repro8x_baseline_step0300_15s_seed0_official_all6_no_tf"

echo ALL_REPRO8X_STEP300_OFFICIAL_QUALITY6_COMPLETE
