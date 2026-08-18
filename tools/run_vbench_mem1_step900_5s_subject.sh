#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/Causal-Forcing
VBENCH_ROOT=/root/autodl-tmp/VBench
GEN_PY=/root/autodl-tmp/envs/longlive/bin/python
EVAL_PY=/root/autodl-tmp/envs/vbench/bin/python
CONFIG="$BASE/configs/causal_forcing_dmd_chunkwise_senior_mem1_nonorm.yaml"
CHECKPOINT="$BASE/logsv1/repro4x_accum2_causal_forcing_dmd_seed_mem1_nonorm_chunkwise/checkpoint_model_000900/model.pt"
MANIFEST="$BASE/prompts/vbench_quality_no_tf_seed42.json"
VIDEOS="$BASE/outputsv1/vbench_mem1_step0900_vbench5s_seed0_subject"
RESULTS="$BASE/outputsv1/vbench_results_mem1_step0900_vbench5s_seed0_subject"

mkdir -p "$VIDEOS" "$RESULTS"
cd "$BASE"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export PYTHONPATH="$BASE:$BASE/long_video"

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
  --suite subject_consistency \
  --use_ema

count=$(find "$VIDEOS" -maxdepth 1 -type f -name '*.mp4' -size +0c | wc -l)
if [[ "$count" -ne 360 ]]; then
  echo "Expected 360 videos, found $count" >&2
  exit 1
fi

cd "$VBENCH_ROOT"
export PYTHONPATH="$VBENCH_ROOT"
"$EVAL_PY" -m torch.distributed.run --standalone --nproc_per_node=4 evaluate.py \
  --videos_path "$VIDEOS" \
  --output_path "$RESULTS" \
  --dimension subject_consistency \
  --load_ckpt_from_local True \
  --mode vbench_standard

touch "$RESULTS/COMPLETE"
