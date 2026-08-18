#!/usr/bin/env bash
set -euo pipefail

MODEL_KIND=${1:?usage: $0 baseline|mem1_nonorm GPU_LIST}
GPU_LIST=${2:?usage: $0 baseline|mem1_nonorm GPU_LIST}

BASE=${BASE:-/root/autodl-tmp/Causal-Forcing}
VBENCH_ROOT=${VBENCH_ROOT:-/root/autodl-tmp/VBench}
GEN_PY=${GEN_PY:-/root/autodl-tmp/envs/longlive/bin/python}
EVAL_PY=${EVAL_PY:-/root/autodl-tmp/envs/vbench/bin/python}
MANIFEST="$BASE/prompts/vbench_quality_no_tf_seed42.json"

case "$MODEL_KIND" in
  baseline)
    CONFIG="$BASE/configs/causal_forcing_dmd_chunkwise_senior_baseline.yaml"
    CHECKPOINT="$BASE/logsv1/repro8x_causal_forcing_dmd_seed_baseline_step300_chunkwise/checkpoint_model_000300/model.pt"
    VIDEOS="$BASE/outputsv1/vbench_repro8x_baseline_step0300_15s_seed0_all6"
    OUT="$BASE/outputsv1/vbench_results_repro8x_baseline_step0300_15s_seed0_no_tf"
    ;;
  mem1_nonorm)
    CONFIG="$BASE/configs/causal_forcing_dmd_chunkwise_senior_mem1_nonorm.yaml"
    CHECKPOINT="$BASE/logsv1/repro8x_causal_forcing_dmd_seed_mem1_nonorm_step300_chunkwise/checkpoint_model_000300/model.pt"
    VIDEOS="$BASE/outputsv1/vbench_repro8x_mem1_step0300_15s_seed0_all6"
    OUT="$BASE/outputsv1/vbench_results_repro8x_mem1_step0300_15s_seed0_no_tf"
    ;;
  *)
    echo "unknown model kind: $MODEL_KIND" >&2
    exit 2
    ;;
esac

DIMS=(
  subject_consistency
  motion_smoothness
  dynamic_degree
  background_consistency
  aesthetic_quality
  imaging_quality
)

[[ -s "$CHECKPOINT" ]] || { echo "MISSING_CHECKPOINT:$CHECKPOINT" >&2; exit 1; }
mkdir -p "$VIDEOS" "$OUT/by_dimension"

export CUDA_VISIBLE_DEVICES="$GPU_LIST"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

cd "$BASE"
export PYTHONPATH="$BASE:$BASE/long_video"
echo "START_GENERATION:$MODEL_KIND:step0300:63_latent_249_pixel_frames:gpus=$GPU_LIST"
"$GEN_PY" -m torch.distributed.run --standalone --nproc_per_node=4 \
  tools/vbench_generate.py \
  --config_path "$CONFIG" \
  --checkpoint_path "$CHECKPOINT" \
  --manifest "$MANIFEST" \
  --output_folder "$VIDEOS" \
  --num_output_frames 63 \
  --samples_per_prompt 5 \
  --batch_size 4 \
  --base_seed 0 \
  --seed_mode prompt_batch_stream \
  --rank_seed_offset \
  --use_ema

count=$(find "$VIDEOS" -maxdepth 1 -type f -name '*.mp4' -size +0c | wc -l)
if [[ "$count" -ne 1255 ]]; then
  echo "EXPECTED_1255_VIDEOS:found${count}" >&2
  exit 1
fi
echo "GENERATION_COMPLETE:$MODEL_KIND:step0300:15s:1255"

cd "$VBENCH_ROOT"
export PYTHONPATH="$VBENCH_ROOT"
for dim in "${DIMS[@]}"; do
  dim_out="$OUT/by_dimension/$dim"
  mkdir -p "$dim_out"
  existing=$(find "$dim_out" -maxdepth 1 -type f -name '*_eval_results.json' -print -quit)
  if [[ -n "$existing" ]]; then
    echo "SKIP_COMPLETE:$dim:$existing"
    continue
  fi
  echo "START_DIMENSION:$MODEL_KIND:$dim"
  "$EVAL_PY" -m torch.distributed.run --standalone --nproc_per_node=4 evaluate.py \
    --videos_path "$VIDEOS" \
    --output_path "$dim_out" \
    --dimension "$dim" \
    --load_ckpt_from_local True \
    --mode vbench_standard
  result=$(find "$dim_out" -maxdepth 1 -type f -name '*_eval_results.json' -print -quit)
  [[ -n "$result" ]] || { echo "MISSING_RESULT:$MODEL_KIND:$dim" >&2; exit 1; }
  echo "DONE_DIMENSION:$MODEL_KIND:$dim:$result"
done

mapfile -t result_files < <(find "$OUT/by_dimension" -mindepth 2 -maxdepth 2 \
  -type f -name '*_eval_results.json' | sort)
[[ "${#result_files[@]}" -eq 6 ]] || {
  echo "EXPECTED_6_RESULTS:found${#result_files[@]}" >&2
  exit 1
}
"$EVAL_PY" "$BASE/tools/merge_vbench_dimension_results.py" \
  --output "$OUT/combined_eval_results.json" "${result_files[@]}"
"$EVAL_PY" "$BASE/tools/calc_vbench_quality_no_tf.py" \
  "$OUT/combined_eval_results.json" > "$OUT/quality_summary_no_tf.json"
touch "$OUT/COMPLETE"
echo "ALL_STEP0300_15S_DIMENSIONS_COMPLETE:$MODEL_KIND"
