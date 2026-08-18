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

run_subject() {
  local kind="$1"
  local config="$2"
  local checkpoint="$3"
  local videos="$4"
  local out="$5"

  [[ -s "$checkpoint" ]] || { echo "MISSING_CHECKPOINT:$checkpoint" >&2; exit 1; }
  mkdir -p "$videos" "$out"

  cd "$BASE"
  export PYTHONPATH="$BASE:$BASE/long_video"
  echo "START_SUBJECT_GENERATION:$kind:step0300:8gpu:base_seed0:63_latent"
  "$GEN_PY" -m torch.distributed.run --standalone --nproc_per_node=8 \
    tools/vbench_generate.py \
    --config_path "$config" \
    --checkpoint_path "$checkpoint" \
    --manifest "$MANIFEST" \
    --output_folder "$videos" \
    --suite subject_consistency \
    --num_output_frames 63 \
    --samples_per_prompt 5 \
    --batch_size 4 \
    --base_seed 0 \
    --seed_mode prompt_batch_stream \
    --rank_seed_offset \
    --use_ema

  local count
  count=$(find "$videos" -maxdepth 1 -type f -name '*.mp4' -size +0c | wc -l)
  [[ "$count" -eq 360 ]] || { echo "EXPECTED_360_SUBJECT_VIDEOS:found${count}" >&2; exit 1; }
  echo "SUBJECT_GENERATION_COMPLETE:$kind:360"

  cd "$VBENCH_ROOT"
  export PYTHONPATH="$VBENCH_ROOT"
  echo "START_SUBJECT_EVAL:$kind"
  "$EVAL_PY" -m torch.distributed.run --standalone --nproc_per_node=8 evaluate.py \
    --videos_path "$videos" \
    --output_path "$out" \
    --dimension subject_consistency \
    --load_ckpt_from_local True \
    --mode vbench_standard
  local result
  result=$(find "$out" -maxdepth 1 -type f -name '*_eval_results.json' -print -quit)
  [[ -n "$result" ]] || { echo "MISSING_SUBJECT_RESULT:$kind" >&2; exit 1; }
  touch "$out/COMPLETE"
  echo "SUBJECT_EVAL_COMPLETE:$kind:$result"
}

run_subject \
  mem1_nonorm \
  "$BASE/configs/causal_forcing_dmd_chunkwise_senior_mem1_nonorm.yaml" \
  "$BASE/logsv1/repro8x_causal_forcing_dmd_seed_mem1_nonorm_step300_chunkwise/checkpoint_model_000300/model.pt" \
  "$BASE/outputsv1/vbench_repro8x_mem1_step0300_15s_seed0_subject" \
  "$BASE/outputsv1/vbench_results_repro8x_mem1_step0300_15s_seed0_subject"

run_subject \
  baseline \
  "$BASE/configs/causal_forcing_dmd_chunkwise_senior_baseline.yaml" \
  "$BASE/logsv1/repro8x_causal_forcing_dmd_seed_baseline_step300_chunkwise/checkpoint_model_000300/model.pt" \
  "$BASE/outputsv1/vbench_repro8x_baseline_step0300_15s_seed0_subject" \
  "$BASE/outputsv1/vbench_results_repro8x_baseline_step0300_15s_seed0_subject"

echo ALL_REPRO8X_STEP300_SUBJECT_COMPLETE
