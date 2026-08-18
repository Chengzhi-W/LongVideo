#!/usr/bin/env bash
set -euo pipefail

BASE=${BASE:-/root/autodl-tmp/Causal-Forcing}
VBENCH_ROOT=${VBENCH_ROOT:-/root/autodl-tmp/VBench}
GEN_PY=${GEN_PY:-/root/autodl-tmp/envs/longlive/bin/python}
EVAL_PY=${EVAL_PY:-/root/autodl-tmp/envs/vbench/bin/python}

CONFIG="$BASE/configs/causal_forcing_dmd_chunkwise_long_vs_local_joint_t750.yaml"
CHECKPOINT="$BASE/logsv1/repro8x_long_vs_local_joint_t750_step300_seed42/checkpoint_model_000300/model.pt"
MANIFEST="$BASE/prompts/vbench_quality_no_tf_seed42.json"
VIDEOS="$BASE/outputsv1/vbench_lrl_t750_step0300_15s_seed0_quality6"
RESULTS="$BASE/outputsv1/vbench_results_lrl_t750_step0300_15s_seed0_quality6_no_tf"

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
  find "$VIDEOS" -maxdepth 1 -type f -name '*.mp4' -size +0c 2>/dev/null | wc -l
}

[[ -s "$CONFIG" ]] || { echo "MISSING_CONFIG:$CONFIG" >&2; exit 1; }
[[ -s "$CHECKPOINT" ]] || { echo "MISSING_CHECKPOINT:$CHECKPOINT" >&2; exit 1; }
[[ -s "$MANIFEST" ]] || { echo "MISSING_MANIFEST:$MANIFEST" >&2; exit 1; }
mkdir -p "$VIDEOS" "$RESULTS/by_dimension"

# Locked preflight: the manifest must contain exactly the three official suites
# whose prompt sets back all six requested VBench dimensions.
"$GEN_PY" - "$MANIFEST" <<'PY'
import json, sys
from collections import Counter

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
counts = Counter(item["suite"] for item in manifest)
expected = {
    "subject_consistency": 72,
    "scene": 86,
    "overall_consistency": 93,
}
assert counts == expected, (counts, expected)
assert len(manifest) == 251
assert len({item["prompt"] for item in manifest}) == 251
print(f"MANIFEST_AUDIT:{dict(counts)}:videos={len(manifest) * 5}")
PY

count=$(video_count)
if [[ "$count" -lt 1255 ]]; then
  cd "$BASE"
  export PYTHONPATH="$BASE:$BASE/long_video"
  echo "START_GENERATION:lrl_t750:step0300:8gpu:batch4:strict_seed0:63_latent:1255"
  "$GEN_PY" -m torch.distributed.run --standalone --nproc_per_node=8 \
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
    --use_ema
fi

count=$(video_count)
[[ "$count" -eq 1255 ]] || {
  echo "EXPECTED_1255_VALID_VIDEOS:found${count}" >&2
  exit 1
}
echo "ALL_VIDEOS_READY:1255"

# Check every video, not merely a sample: 63 Wan latent frames decode to 249
# pixel frames, written at 16 fps.  Keep invalid files for diagnosis.
"$GEN_PY" - "$VIDEOS" <<'PY'
import cv2, glob, os, sys

paths = sorted(glob.glob(os.path.join(sys.argv[1], "*.mp4")))
bad = []
for path in paths:
    cap = cv2.VideoCapture(path)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if frames != 249 or abs(fps - 16.0) > 0.01:
        bad.append((os.path.basename(path), frames, fps))
assert len(paths) == 1255, len(paths)
assert not bad, bad[:20]
print("VIDEO_AUDIT:count=1255:frames=249:fps=16:duration=15.5625s")
PY

for dim in "${DIMS[@]}"; do
  dim_out="$RESULTS/by_dimension/$dim"
  mkdir -p "$dim_out"
  existing=$(find "$dim_out" -maxdepth 1 -type f -name '*_eval_results.json' -print -quit)
  if [[ -n "$existing" ]]; then
    echo "SKIP_COMPLETE:$dim:$existing"
    continue
  fi

  cd "$VBENCH_ROOT"
  export PYTHONPATH="$VBENCH_ROOT"
  echo "START_DIMENSION:$dim:8gpu"
  "$EVAL_PY" -m torch.distributed.run --standalone --nproc_per_node=8 evaluate.py \
    --videos_path "$VIDEOS" \
    --output_path "$dim_out" \
    --dimension "$dim" \
    --load_ckpt_from_local True \
    --mode vbench_standard

  result=$(find "$dim_out" -maxdepth 1 -type f -name '*_eval_results.json' -print -quit)
  [[ -n "$result" ]] || { echo "MISSING_RESULT:$dim" >&2; exit 1; }
  echo "DONE_DIMENSION:$dim:$result"
done

mapfile -t result_files < <(find "$RESULTS/by_dimension" -mindepth 2 -maxdepth 2 \
  -type f -name '*_eval_results.json' | sort)
[[ "${#result_files[@]}" -eq 6 ]] || {
  echo "EXPECTED_6_RESULTS:found${#result_files[@]}" >&2
  exit 1
}
if printf '%s\n' "${result_files[@]}" | grep -q temporal_flickering; then
  echo "TEMPORAL_FLICKERING_MUST_NOT_BE_INCLUDED" >&2
  exit 1
fi

"$EVAL_PY" "$BASE/tools/merge_vbench_dimension_results.py" \
  --output "$RESULTS/combined_eval_results.json" "${result_files[@]}"
"$EVAL_PY" "$BASE/tools/calc_vbench_quality_no_tf.py" \
  "$RESULTS/combined_eval_results.json" > "$RESULTS/quality_summary_no_tf.json"
touch "$RESULTS/COMPLETE"
echo "LRL_T750_STEP300_QUALITY6_COMPLETE:$RESULTS"
