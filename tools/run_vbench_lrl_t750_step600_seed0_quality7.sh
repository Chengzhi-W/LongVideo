#!/usr/bin/env bash
set -euo pipefail

BASE=${BASE:-/root/autodl-tmp/Causal-Forcing}
VBENCH_ROOT=${VBENCH_ROOT:-/root/autodl-tmp/VBench}
GEN_PY=${GEN_PY:-/root/autodl-tmp/envs/longlive/bin/python}
EVAL_PY=${EVAL_PY:-/root/autodl-tmp/envs/vbench/bin/python}

CONFIG="$BASE/configs/causal_forcing_dmd_chunkwise_long_vs_local_joint_t750_openended.yaml"
CHECKPOINT="$BASE/logsv1/repro8x_long_vs_local_joint_t750_openended_seed42/checkpoint_model_000600/model.pt"
QUALITY_MANIFEST="$BASE/prompts/vbench_quality_no_tf_seed42.json"
TF_MANIFEST="$BASE/prompts/vbench_temporal_flickering_seed42.json"
PREFIX="vbench_lrl_t750_step0600_15s_allrankseed0"

SUBJECT_VIDEOS="$BASE/outputsv1/${PREFIX}_subject"
TF_VIDEOS="$BASE/outputsv1/${PREFIX}_temporal_flickering"
SCENE_VIDEOS="$BASE/outputsv1/${PREFIX}_scene"
OVERALL_VIDEOS="$BASE/outputsv1/${PREFIX}_overall"
SUBJECT_RESULTS="$BASE/outputsv1/${PREFIX}_subject3_results"
TF_RESULTS="$BASE/outputsv1/${PREFIX}_temporal_flickering_results"
REMAINING_RESULTS="$BASE/outputsv1/${PREFIX}_remaining3_results"
QUALITY7_RESULTS="$BASE/outputsv1/${PREFIX}_quality7_results"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUNBUFFERED=1
export PYTHONPATH="$BASE:$BASE/long_video"

for path in "$CONFIG" "$CHECKPOINT" "$QUALITY_MANIFEST" "$TF_MANIFEST"; do
  [[ -s "$path" ]] || { echo "MISSING_REQUIRED_FILE:$path" >&2; exit 1; }
done

"$GEN_PY" - "$CONFIG" "$QUALITY_MANIFEST" "$TF_MANIFEST" <<'PY'
import json, sys
from collections import Counter
from omegaconf import OmegaConf

c = OmegaConf.load(sys.argv[1])
quality = json.load(open(sys.argv[2], encoding="utf-8"))
tf = json.load(open(sys.argv[3], encoding="utf-8"))
assert c.seed == 42
assert c.use_memory_dmd is True
assert c.memory_control_type == "local_history"
assert c.memory_full_frames == 18 and c.memory_local_frames == 6
assert c.memory_max_timestep == 750
assert c.memory_joint_normalization is True
assert c.memory_dmd_scale == 1.0
assert c.memory_sink_frames == 0
assert c.memory_highpass is False
assert c.batch_size == 1 and c.total_batch_size == 64
assert Counter(x["suite"] for x in quality) == {
    "subject_consistency": 72, "scene": 86, "overall_consistency": 93}
assert len(tf) == 75 and all(x["suite"] == "temporal_flickering" for x in tf)
print("CONFIG_AUDIT:A18-A6:t<750:joint_norm:scale1:step600_EMA")
print("PROTOCOL_AUDIT:15s:63latent:249frames:8gpu:batch4:all_rank_seed0")
print("MANIFEST_AUDIT:subject72:tf75:scene86:overall93:samples5")
PY

video_count() {
  find "$1" -maxdepth 1 -type f -name '*.mp4' -size +10k 2>/dev/null | wc -l
}

generate_suite() {
  local suite=$1 expected=$2 manifest=$3 videos=$4
  mkdir -p "$videos"
  local count
  count=$(video_count "$videos")
  if [[ "$count" -lt "$expected" ]]; then
    cd "$BASE"
    # evaluate_dim switches PYTHONPATH to the VBench checkout. Restore the
    # generation environment on every suite transition so later suites can
    # import Causal-Forcing helpers such as demo_utils.
    export PYTHONPATH="$BASE:$BASE/long_video"
    export OMP_NUM_THREADS=8
    echo "START_GENERATION:$suite:expected${expected}:8gpu:batch4:all_rank_seed0"
    "$GEN_PY" -m torch.distributed.run --standalone --nproc_per_node=8 \
      tools/vbench_generate.py \
      --config_path "$CONFIG" \
      --checkpoint_path "$CHECKPOINT" \
      --manifest "$manifest" \
      --output_folder "$videos" \
      --suite "$suite" \
      --num_output_frames 63 \
      --samples_per_prompt 5 \
      --batch_size 4 \
      --base_seed 0 \
      --seed_mode prompt_batch_stream \
      --use_ema
  fi
  count=$(video_count "$videos")
  [[ "$count" -eq "$expected" ]] || {
    echo "VIDEO_COUNT_MISMATCH:$suite:expected${expected}:found${count}" >&2
    exit 1
  }
  "$GEN_PY" - "$videos" "$expected" <<'PY'
import cv2, glob, os, sys
paths = sorted(glob.glob(os.path.join(sys.argv[1], "*.mp4")))
expected = int(sys.argv[2])
bad = []
for path in paths:
    cap = cv2.VideoCapture(path)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if frames != 249 or abs(fps - 16.0) > 0.01:
        bad.append((os.path.basename(path), frames, fps))
assert len(paths) == expected, (len(paths), expected)
assert not bad, bad[:20]
print(f"VIDEO_AUDIT:count={expected}:frames=249:fps=16:duration=15.5625s")
PY
}

evaluate_dim() {
  local videos=$1 dim=$2 out=$3
  mkdir -p "$out"
  local existing
  existing=$(find "$out" -maxdepth 1 -type f -name '*_eval_results.json' -print -quit)
  if [[ -n "$existing" ]]; then
    echo "SKIP_COMPLETE:$dim:$existing"
    return
  fi
  cd "$VBENCH_ROOT"
  export PYTHONPATH="$VBENCH_ROOT"
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
  export VBENCH_TORCH_THREADS=1
  echo "START_DIMENSION:$dim:8gpu"
  "$EVAL_PY" -m torch.distributed.run --standalone --nproc_per_node=8 evaluate.py \
    --videos_path "$videos" \
    --output_path "$out" \
    --dimension "$dim" \
    --load_ckpt_from_local True \
    --mode vbench_standard
  existing=$(find "$out" -maxdepth 1 -type f -name '*_eval_results.json' -print -quit)
  [[ -n "$existing" ]] || { echo "MISSING_RESULT:$dim" >&2; exit 1; }
  echo "DONE_DIMENSION:$dim:$existing"
}

generate_suite subject_consistency 360 "$QUALITY_MANIFEST" "$SUBJECT_VIDEOS"
for dim in subject_consistency dynamic_degree motion_smoothness; do
  evaluate_dim "$SUBJECT_VIDEOS" "$dim" "$SUBJECT_RESULTS/by_dimension/$dim"
done
mapfile -t subject_files < <(find "$SUBJECT_RESULTS/by_dimension" -mindepth 2 -maxdepth 2 \
  -type f -name '*_eval_results.json' | sort)
[[ "${#subject_files[@]}" -eq 3 ]]
"$EVAL_PY" "$BASE/tools/merge_vbench_dimension_results.py" \
  --output "$SUBJECT_RESULTS/combined_eval_results.json" \
  --expected-dimensions subject_consistency,dynamic_degree,motion_smoothness \
  "${subject_files[@]}"
touch "$SUBJECT_RESULTS/COMPLETE"
echo "SUBJECT3_COMPLETE"

generate_suite temporal_flickering 375 "$TF_MANIFEST" "$TF_VIDEOS"
evaluate_dim "$TF_VIDEOS" temporal_flickering "$TF_RESULTS"
touch "$TF_RESULTS/COMPLETE"
echo "TEMPORAL_FLICKERING_COMPLETE"

generate_suite scene 430 "$QUALITY_MANIFEST" "$SCENE_VIDEOS"
evaluate_dim "$SCENE_VIDEOS" background_consistency \
  "$REMAINING_RESULTS/by_dimension/background_consistency"

generate_suite overall_consistency 465 "$QUALITY_MANIFEST" "$OVERALL_VIDEOS"
evaluate_dim "$OVERALL_VIDEOS" aesthetic_quality \
  "$REMAINING_RESULTS/by_dimension/aesthetic_quality"
evaluate_dim "$OVERALL_VIDEOS" imaging_quality \
  "$REMAINING_RESULTS/by_dimension/imaging_quality"

mkdir -p "$QUALITY7_RESULTS"
"$EVAL_PY" - "$SUBJECT_RESULTS" "$TF_RESULTS" "$REMAINING_RESULTS" "$QUALITY7_RESULTS" <<'PY'
import glob, json, os, sys
from pathlib import Path

subject3, tf_root, remaining, out = sys.argv[1:]
bounds = {
    "subject_consistency": (0.1462, 1.0),
    "background_consistency": (0.2615, 1.0),
    "temporal_flickering": (0.6293, 1.0),
    "motion_smoothness": (0.7060, 0.9975),
    "aesthetic_quality": (0.0, 1.0),
    "imaging_quality": (0.0, 1.0),
    "dynamic_degree": (0.0, 1.0),
}
raw = {}
for pattern in [
    os.path.join(subject3, "combined_eval_results.json"),
    os.path.join(tf_root, "*_eval_results.json"),
    os.path.join(remaining, "by_dimension", "*", "*_eval_results.json"),
]:
    for path in glob.glob(pattern):
        data = json.load(open(path, encoding="utf-8"))
        for key, value in data.items():
            if key in bounds:
                raw[key] = float(value[0] if isinstance(value, list) else value)
missing = sorted(set(bounds) - set(raw))
assert not missing, missing
normalized = {k: (raw[k] - lo) / (hi - lo) for k, (lo, hi) in bounds.items()}
weights = {k: (0.5 if k == "dynamic_degree" else 1.0) for k in bounds}
score7 = sum(normalized[k] * weights[k] for k in bounds) / sum(weights.values())
six = [k for k in bounds if k != "temporal_flickering"]
score6 = sum(normalized[k] * weights[k] for k in six) / sum(weights[k] for k in six)
payload = {
    "label": "lrl_t750_step600", "raw": raw, "normalized": normalized,
    "weights": weights, "quality_score_no_tf": score6,
    "quality_score_all7": score7,
}
Path(out, "quality_summary_all7.json").write_text(
    json.dumps(payload, indent=2), encoding="utf-8")
print(f"QUALITY6:{score6 * 100:.8f}")
print(f"QUALITY7:{score7 * 100:.8f}")
PY
touch "$QUALITY7_RESULTS/COMPLETE"
echo "LRL_T750_STEP600_QUALITY7_COMPLETE"
