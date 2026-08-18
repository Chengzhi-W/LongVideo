#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-$ROOT/dataset/UltraVideo-Long-5K}"
MANIFEST_IDS="$ROOT/data_manifests/ultravideo_long_5k/selection_ids.csv"
FULL_MANIFEST="$DATA_ROOT/manifests/train.csv"
ACTION="${1:-help}"

download_args=(
  --output-dir "$DATA_ROOT"
  --selection-ids "$MANIFEST_IDS"
  --count 5000
  --seed 42
  --resolution 960
  --min-duration 15.6
  --val-count 100
)

case "$ACTION" in
  metadata)
    "$PYTHON" "$ROOT/tools/download_ultravideo_long_5k.py" \
      "${download_args[@]}" --metadata-only
    ;;
  download)
    "$PYTHON" "$ROOT/tools/download_ultravideo_long_5k.py" \
      "${download_args[@]}"
    ;;
  debug-download)
    "$PYTHON" "$ROOT/tools/download_ultravideo_long_debug_subset.py" \
      --output-dir "$ROOT/dataset/UltraVideo-Long-debug" \
      --selection-ids "$MANIFEST_IDS" \
      --count "${DEBUG_COUNT:-4}" --offset "${DEBUG_OFFSET:-0}" \
      --resolution 960 --min-duration 15.6
    ;;
  debug-encode)
    "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=1 \
      "$ROOT/tools/prepare_ultravideo_long_latents.py" \
      --manifest "$ROOT/dataset/UltraVideo-Long-debug/manifests/train.csv" \
      --videos-dir "$ROOT/dataset/UltraVideo-Long-debug/videos" \
      --output-dir "$ROOT/dataset/UltraVideo-Long-debug/latents_63_240x416" \
      --num-frames 249 --fps 16 --height 240 --width 416 --seed 42
    ;;
  debug-build)
    "$PYTHON" "$ROOT/tools/build_ultravideo_long_lmdb.py" \
      --latents-dir "$ROOT/dataset/UltraVideo-Long-debug/latents_63_240x416" \
      --output "$ROOT/dataset/UltraVideo-Long-debug/clean_data_63_240x416" \
      --map-size-gb 64
    ;;
  check)
    "$PYTHON" "$ROOT/tools/prepare_ultravideo_long_latents.py" \
      --manifest "$FULL_MANIFEST" \
      --videos-dir "$DATA_ROOT/videos" \
      --output-dir "$DATA_ROOT/latents_63" \
      --num-frames 249 --limit "${LIMIT:-8}" --dry-run
    ;;
  encode-long)
    "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE:-8}" \
      "$ROOT/tools/prepare_ultravideo_long_latents.py" \
      --manifest "$FULL_MANIFEST" \
      --videos-dir "$DATA_ROOT/videos" \
      --output-dir "$DATA_ROOT/latents_63" \
      --num-frames 249 --fps 16 --height 480 --width 832 --seed 42
    ;;
  encode-5s)
    "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE:-8}" \
      "$ROOT/tools/prepare_ultravideo_long_latents.py" \
      --manifest "$FULL_MANIFEST" \
      --videos-dir "$DATA_ROOT/videos" \
      --output-dir "$DATA_ROOT/latents_21" \
      --num-frames 81 --fps 16 --height 480 --width 832 --seed 42
    ;;
  build-long)
    "$PYTHON" "$ROOT/tools/build_ultravideo_long_lmdb.py" \
      --latents-dir "$DATA_ROOT/latents_63" \
      --output "$DATA_ROOT/clean_data_63"
    ;;
  build-5s)
    "$PYTHON" "$ROOT/tools/build_ultravideo_long_lmdb.py" \
      --latents-dir "$DATA_ROOT/latents_21" \
      --output "$DATA_ROOT/clean_data_21"
    ;;
  help|-h|--help)
    cat <<'EOF'
Usage: bash tools/ultravideo_long_5k.sh ACTION

Actions (each is safe to rerun unless noted):
  metadata     Download official metadata and regenerate the locked manifests
  download     Resume official ZIP downloads and extract the locked 5K clips
  debug-download  Range-fetch DEBUG_COUNT=4 selected videos (no full ZIPs)
  debug-encode    Encode 249 frames -> 63 latents at smoke resolution 240x416
  debug-build     Build the smoke LMDB consumed by the one-GPU config
  check        Validate the first LIMIT=8 videos without loading a GPU model
  encode-long  Encode 249 pixel frames -> 63 Wan latent frames on 8 GPUs
  encode-5s    Encode 81 pixel frames -> 21 Wan latent frames on 8 GPUs
  build-long   Atomically build the 63-latent LMDB (must not already exist)
  build-5s     Atomically build the 21-latent LMDB (must not already exist)

Optional environment variables: ROOT, PYTHON, DATA_ROOT, NPROC_PER_NODE, LIMIT.
EOF
    ;;
  *)
    echo "Unknown action: $ACTION" >&2
    exit 2
    ;;
esac
