#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-$ROOT/dataset/UltraVideo-Long-5K-20s}"
MANIFEST_IDS="$ROOT/data_manifests/ultravideo_long_5k_20s/selection_ids.csv"
FULL_MANIFEST="$DATA_ROOT/manifests/train.csv"
ACTION="${1:-help}"

download_args=(
  --output-dir "$DATA_ROOT"
  --selection-ids "$MANIFEST_IDS"
  --count 5000
  --seed 42
  --resolution 960
  --min-duration 20.0
  --val-count 100
  --metadata-sha256 6a14434c3c74b79a5ef3c8bf5f60fbd1b5f71cb27c1faea9300d0e372e961191
)

case "$ACTION" in
  metadata|metadata-20s)
    "$PYTHON" "$ROOT/tools/download_ultravideo_long_5k.py" \
      "${download_args[@]}" --metadata-only
    ;;
  download|download-20s)
    "$PYTHON" "$ROOT/tools/download_ultravideo_long_5k.py" \
      "${download_args[@]}"
    ;;
  debug-download)
    "$PYTHON" "$ROOT/tools/download_ultravideo_long_debug_subset.py" \
      --output-dir "$ROOT/dataset/UltraVideo-Long-debug" \
      --selection-ids "$MANIFEST_IDS" \
      --count "${DEBUG_COUNT:-4}" --offset "${DEBUG_OFFSET:-0}" \
      --resolution 960 --min-duration 20.0
    ;;
  debug-encode-20s)
    "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=1 \
      "$ROOT/tools/prepare_ultravideo_long_latents.py" \
      --manifest "$ROOT/dataset/UltraVideo-Long-debug/manifests/train.csv" \
      --videos-dir "$ROOT/dataset/UltraVideo-Long-debug/videos" \
      --output-dir "$ROOT/dataset/UltraVideo-Long-debug/latents_81_240x416" \
      --num-frames 321 --fps 16 --height 240 --width 416 --seed 42 \
      --skip-too-short
    ;;
  debug-build-20s)
    "$PYTHON" "$ROOT/tools/build_ultravideo_long_lmdb.py" \
      --latents-dir "$ROOT/dataset/UltraVideo-Long-debug/latents_81_240x416" \
      --output "$ROOT/dataset/UltraVideo-Long-debug/clean_data_81_240x416" \
      --map-size-gb 64 --require-complete
    ;;
  debug-audit-20s)
    audit_args=(
      --manifest "$ROOT/dataset/UltraVideo-Long-debug/manifests/train.csv"
      --videos-dir "$ROOT/dataset/UltraVideo-Long-debug/videos"
      --latents-dir "$ROOT/dataset/UltraVideo-Long-debug/latents_81_240x416"
      --lmdb "$ROOT/dataset/UltraVideo-Long-debug/clean_data_81_240x416"
      --expected-count 3
      --latent-shape 81 16 30 52
    )
    [[ "${FULL_AUDIT:-0}" == 1 ]] && audit_args+=(--full)
    "$PYTHON" "$ROOT/tools/audit_ultravideo_long_20s_data.py" \
      "${audit_args[@]}"
    ;;
  check|check-20s)
    "$PYTHON" "$ROOT/tools/prepare_ultravideo_long_latents.py" \
      --manifest "$FULL_MANIFEST" \
      --videos-dir "$DATA_ROOT/videos" \
      --output-dir "$DATA_ROOT/latents_81" \
      --num-frames 321 --fps 16 --height 480 --width 832 --seed 42 \
      --skip-too-short --limit "${LIMIT:-0}" --dry-run
    ;;
  encode-20s)
    "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE:-8}" \
      "$ROOT/tools/prepare_ultravideo_long_latents.py" \
      --manifest "$FULL_MANIFEST" \
      --videos-dir "$DATA_ROOT/videos" \
      --output-dir "$DATA_ROOT/latents_81" \
      --num-frames 321 --fps 16 --height 480 --width 832 --seed 42 \
      --skip-too-short
    ;;
  build-20s)
    "$PYTHON" "$ROOT/tools/build_ultravideo_long_lmdb.py" \
      --latents-dir "$DATA_ROOT/latents_81" \
      --output "$DATA_ROOT/clean_data_81" \
      --map-size-gb 128 --require-complete
    ;;
  audit-20s)
    audit_args=(
      --manifest "$FULL_MANIFEST"
      --videos-dir "$DATA_ROOT/videos"
      --latents-dir "$DATA_ROOT/latents_81"
      --lmdb "$DATA_ROOT/clean_data_81"
      --expected-count 4900
      --latent-shape 81 16 60 104
    )
    [[ "${FULL_AUDIT:-0}" == 1 ]] && audit_args+=(--full)
    "$PYTHON" "$ROOT/tools/audit_ultravideo_long_20s_data.py" \
      "${audit_args[@]}"
    ;;
  help|-h|--help)
    cat <<'EOF'
Usage: bash tools/ultravideo_long_5k.sh ACTION

Actions (each is safe to rerun unless noted):
  metadata       Build manifests for the locked 5K clips that are all >=20s
  download       Download/extract the locked 5K-20s videos (requires internet)
  metadata-20s   Explicit alias for metadata
  download-20s   Explicit alias for download
  debug-download  Range-fetch DEBUG_COUNT=4 selected videos (no full ZIPs)
  debug-encode-20s  Encode all >=20s debug clips: 321 frames -> 81 latents
  debug-build-20s   Build the 81-latent smoke LMDB
  debug-audit-20s   Audit the debug videos, shards and LMDB end to end
  check        Alias for check-20s
  check-20s    Validate all 4,900 training videos; set LIMIT=N for a prefix
  encode-20s   Encode >=20s clips: 321 pixel frames -> 81 Wan latents
  build-20s    Atomically build the 81-latent LMDB (must not already exist)
  audit-20s    Audit manifest/shards/LMDB; set FULL_AUDIT=1 for every item

Optional environment variables: ROOT, PYTHON, DATA_ROOT, NPROC_PER_NODE, LIMIT,
FULL_AUDIT.
EOF
    ;;
  *)
    echo "Unknown action: $ACTION" >&2
    exit 2
    ;;
esac
