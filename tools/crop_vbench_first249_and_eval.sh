#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/Causal-Forcing
SRC="$BASE/outputsv1/vbench_mem1_step0900_aligned_seed0_126_subject"
DST="$BASE/outputsv1/vbench_mem1_step0900_aligned_seed0_first249_subject"
OUT="$BASE/outputsv1/vbench_results_mem1_step0900_aligned_seed0_first249_subject"
FFMPEG=/root/autodl-tmp/envs/longlive/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2

mkdir -p "$DST" "$OUT"
export DST FFMPEG
find "$SRC" -type f -name '*.mp4' -print0 |
  xargs -0 -P 16 -I{} bash -c '
    output="$DST/$(basename "$1")"
    if [[ ! -s "$output" ]]; then
      "$FFMPEG" -v error -i "$1" -frames:v 249 -c:v copy -an "$output"
    fi
  ' _ {}

count=$(find "$DST" -type f -name '*.mp4' -size +0c | wc -l)
if [[ "$count" -ne 360 ]]; then
  echo "Expected 360 cropped videos, found $count" >&2
  exit 1
fi
touch "$DST/CROP_COMPLETE"

cd /root/autodl-tmp/VBench
export PYTHONPATH=/root/autodl-tmp/VBench
export CUDA_VISIBLE_DEVICES=0,1,2,3
/root/autodl-tmp/envs/vbench/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=4 evaluate.py \
  --videos_path "$DST" \
  --output_path "$OUT" \
  --dimension subject_consistency \
  --load_ckpt_from_local True \
  --mode vbench_standard
touch "$OUT/SUBJECT_COMPLETE"
