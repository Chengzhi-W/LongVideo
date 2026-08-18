#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/Causal-Forcing
VBENCH_ROOT=/root/autodl-tmp/VBench
GEN_PY=/root/autodl-tmp/envs/longlive/bin/python
EVAL_PY=/root/autodl-tmp/envs/vbench/bin/python
CONFIG="$BASE/configs/causal_forcing_dmd_chunkwise_senior_mem1_nonorm.yaml"
CHECKPOINT="$BASE/logsv1/repro4x_accum2_causal_forcing_dmd_seed_mem1_nonorm_chunkwise/checkpoint_model_000900/model.pt"
MANIFEST="$BASE/prompts/vbench_quality_no_tf_seed42.json"
SUBJECT_VIDEOS="$BASE/outputsv1/vbench_mem1_step0900_vbench5s_seed0_subject"
VIDEOS="$BASE/outputsv1/vbench_mem1_step0900_vbench5s_seed0_all6"
SUBJECT_RESULTS="$BASE/outputsv1/vbench_results_mem1_step0900_vbench5s_seed0_subject"
OUT="$BASE/outputsv1/vbench_results_mem1_step0900_vbench5s_seed0_no_tf"

mkdir -p "$VIDEOS" "$OUT/by_dimension/subject_consistency"

# Reuse the already validated 360 subject videos without consuming extra disk space.
find "$SUBJECT_VIDEOS" -maxdepth 1 -type f -name '*.mp4' -size +0c -print0 \
  | xargs -0 -I{} cp -al "{}" "$VIDEOS/"

subject_json=$(find "$SUBJECT_RESULTS" -maxdepth 1 -type f -name '*_eval_results.json' -print -quit)
[[ -n "$subject_json" ]] || { echo "Missing completed subject result" >&2; exit 1; }
cp -n "$subject_json" "$OUT/by_dimension/subject_consistency/"

cd "$BASE"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export PYTHONPATH="$BASE:$BASE/long_video"

# The first 72 manifest entries are the already generated subject prompts.  The
# sampler still draws their noise before skipping their files, preserving the exact
# per-rank RNG stream used by an uninterrupted all-dimension run.
"$GEN_PY" -m torch.distributed.run --standalone --nproc_per_node=4 \
  tools/vbench_generate.py \
  --config_path "$CONFIG" \
  --checkpoint_path "$CHECKPOINT" \
  --manifest "$MANIFEST" \
  --output_folder "$VIDEOS" \
  --num_output_frames 21 \
  --samples_per_prompt 5 \
  --batch_size 4 \
  --base_seed 0 \
  --seed_mode prompt_batch_stream \
  --rank_seed_offset \
  --use_ema

count=$(find "$VIDEOS" -maxdepth 1 -type f -name '*.mp4' -size +0c | wc -l)
if [[ "$count" -ne 1255 ]]; then
  echo "Expected 1255 videos, found $count" >&2
  exit 1
fi

cd "$VBENCH_ROOT"
export PYTHONPATH="$VBENCH_ROOT"
dims=(
  background_consistency
  motion_smoothness
  aesthetic_quality
  imaging_quality
  dynamic_degree
)

for dim in "${dims[@]}"; do
  dim_out="$OUT/by_dimension/$dim"
  mkdir -p "$dim_out"
  existing=$(find "$dim_out" -maxdepth 1 -type f -name '*_eval_results.json' -print -quit)
  if [[ -n "$existing" ]]; then
    echo "SKIP_COMPLETE:$dim:$existing"
    continue
  fi
  echo "START_DIMENSION:$dim"
  "$EVAL_PY" -m torch.distributed.run --standalone --nproc_per_node=4 evaluate.py \
    --videos_path "$VIDEOS" \
    --output_path "$dim_out" \
    --dimension "$dim" \
    --load_ckpt_from_local True \
    --mode vbench_standard
  result=$(find "$dim_out" -maxdepth 1 -type f -name '*_eval_results.json' -print -quit)
  [[ -n "$result" ]] || { echo "MISSING_RESULT:$dim" >&2; exit 1; }
  echo "DONE_DIMENSION:$dim:$result"
done

mapfile -t result_files < <(find "$OUT/by_dimension" -mindepth 2 -maxdepth 2 -type f -name '*_eval_results.json' | sort)
[[ "${#result_files[@]}" -eq 6 ]] || { echo "Expected 6 result JSONs, found ${#result_files[@]}" >&2; exit 1; }
"$EVAL_PY" "$BASE/tools/merge_vbench_dimension_results.py" \
  --output "$OUT/combined_eval_results.json" "${result_files[@]}"
"$EVAL_PY" "$BASE/tools/calc_vbench_quality_no_tf.py" \
  "$OUT/combined_eval_results.json" > "$OUT/quality_summary_no_tf.json"
touch "$OUT/COMPLETE"
echo "ALL_SIX_DIMENSIONS_COMPLETE"
