#!/usr/bin/env bash
set -euo pipefail

BASE=${BASE:-/root/autodl-tmp/Causal-Forcing}
VBENCH_ROOT=${VBENCH_ROOT:-/root/autodl-tmp/VBench}
GEN_PY=${GEN_PY:-/root/autodl-tmp/envs/longlive/bin/python}
EVAL_PY=${EVAL_PY:-/root/autodl-tmp/envs/vbench/bin/python}
CONFIG="$BASE/configs/causal_forcing_dmd_chunkwise_senior_baseline.yaml"
CHECKPOINT_ROOT="$BASE/logsv1/repro4x_accum2_causal_forcing_dmd_seed_baseline_chunkwise"
MANIFEST="$BASE/prompts/vbench_quality_no_tf_seed42.json"
STEPS=${STEPS:-"0300 0600 0900 1200 1400"}

DIMS=(
  subject_consistency
  background_consistency
  motion_smoothness
  aesthetic_quality
  imaging_quality
  dynamic_degree
)

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

for short_step in $STEPS; do
  full_step=$(printf "%06d" "$((10#$short_step))")
  checkpoint="$CHECKPOINT_ROOT/checkpoint_model_${full_step}/model.pt"
  videos="$BASE/outputsv1/vbench_baseline_step${short_step}_vbench5s_seed0_all6"
  out="$BASE/outputsv1/vbench_results_baseline_step${short_step}_vbench5s_seed0_no_tf"

  [[ -s "$checkpoint" ]] || { echo "MISSING_CHECKPOINT:$checkpoint" >&2; exit 1; }
  mkdir -p "$videos" "$out/by_dimension"

  if [[ ! -f "$out/COMPLETE" ]]; then
    echo "START_GENERATION:step${short_step}"
    cd "$BASE"
    export PYTHONPATH="$BASE:$BASE/long_video"
    "$GEN_PY" -m torch.distributed.run --standalone --nproc_per_node=4 \
      tools/vbench_generate.py \
      --config_path "$CONFIG" \
      --checkpoint_path "$checkpoint" \
      --manifest "$MANIFEST" \
      --output_folder "$videos" \
      --num_output_frames 21 \
      --samples_per_prompt 5 \
      --batch_size 4 \
      --base_seed 0 \
      --seed_mode prompt_batch_stream \
      --rank_seed_offset \
      --use_ema

    count=$(find "$videos" -maxdepth 1 -type f -name '*.mp4' -size +0c | wc -l)
    if [[ "$count" -ne 1255 ]]; then
      echo "EXPECTED_1255_VIDEOS:step${short_step}:found${count}" >&2
      exit 1
    fi
    echo "GENERATION_COMPLETE:step${short_step}:1255"

    cd "$VBENCH_ROOT"
    export PYTHONPATH="$VBENCH_ROOT"
    for dim in "${DIMS[@]}"; do
      dim_out="$out/by_dimension/$dim"
      mkdir -p "$dim_out"
      existing=$(find "$dim_out" -maxdepth 1 -type f -name '*_eval_results.json' -print -quit)
      if [[ -n "$existing" ]]; then
        echo "SKIP_COMPLETE:step${short_step}:$dim:$existing"
        continue
      fi
      echo "START_DIMENSION:step${short_step}:$dim"
      "$EVAL_PY" -m torch.distributed.run --standalone --nproc_per_node=4 evaluate.py \
        --videos_path "$videos" \
        --output_path "$dim_out" \
        --dimension "$dim" \
        --load_ckpt_from_local True \
        --mode vbench_standard
      result=$(find "$dim_out" -maxdepth 1 -type f -name '*_eval_results.json' -print -quit)
      [[ -n "$result" ]] || { echo "MISSING_RESULT:step${short_step}:$dim" >&2; exit 1; }
      echo "DONE_DIMENSION:step${short_step}:$dim:$result"
    done

    mapfile -t result_files < <(find "$out/by_dimension" -mindepth 2 -maxdepth 2 \
      -type f -name '*_eval_results.json' | sort)
    [[ "${#result_files[@]}" -eq 6 ]] || {
      echo "EXPECTED_6_RESULTS:step${short_step}:found${#result_files[@]}" >&2
      exit 1
    }
    "$EVAL_PY" "$BASE/tools/merge_vbench_dimension_results.py" \
      --output "$out/combined_eval_results.json" "${result_files[@]}"
    "$EVAL_PY" "$BASE/tools/calc_vbench_quality_no_tf.py" \
      "$out/combined_eval_results.json" > "$out/quality_summary_no_tf.json"
    touch "$out/COMPLETE"
  fi
  echo "STEP_COMPLETE:step${short_step}"
done

echo "ALL_BASELINE_STEPS_COMPLETE"
