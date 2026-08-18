#!/usr/bin/env bash
set -euo pipefail

BASE=${BASE:-/root/autodl-tmp/Causal-Forcing}
VBENCH_ROOT=${VBENCH_ROOT:-/root/autodl-tmp/VBench}
GEN_PY=${GEN_PY:-/root/autodl-tmp/envs/longlive/bin/python}
EVAL_PY=${EVAL_PY:-/root/autodl-tmp/envs/vbench/bin/python}

MANIFEST="$BASE/prompts/vbench_quality_no_tf_seed42.json"
METHOD_CONFIG="$BASE/configs/causal_forcing_dmd_chunkwise_long_vs_local_joint_nogate_step1500.yaml"
METHOD_CHECKPOINT="$BASE/logsv1/repro8x_long_vs_local_joint_nogate_step1500_seed42/checkpoint_model_000600/model.pt"
BASELINE_CONFIG="$BASE/configs/causal_forcing_dmd_chunkwise_senior_baseline_step1500.yaml"
BASELINE_CHECKPOINT="$BASE/logsv1/repro8x_causal_forcing_dmd_seed42_baseline_step1500_chunkwise/checkpoint_model_000600/model.pt"

METHOD_TF_RESULTS="$BASE/outputsv1/vbench_results_lrl_nogate_step0600_15s_allrankseed0_temporal_flickering"
BASELINE_TF_RESULTS="$BASE/outputsv1/vbench_results_baseline_step0600_15s_allrankseed0_temporal_flickering"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUNBUFFERED=1

video_count() {
  find "$1" -maxdepth 1 -type f -name '*.mp4' -size +0c 2>/dev/null | wc -l
}

echo "WAIT_OFFICIAL_TF_PAIR_COMPLETE"
until [[ -f "$METHOD_TF_RESULTS/COMPLETE" && -f "$BASELINE_TF_RESULTS/COMPLETE" ]]; do
  sleep 30
done

for path in "$MANIFEST" "$METHOD_CONFIG" "$METHOD_CHECKPOINT" \
  "$BASELINE_CONFIG" "$BASELINE_CHECKPOINT"; do
  [[ -s "$path" ]] || { echo "MISSING_REQUIRED_FILE:$path" >&2; exit 1; }
done

"$GEN_PY" - "$METHOD_CONFIG" "$BASELINE_CONFIG" "$MANIFEST" <<'PY'
import json, sys
from omegaconf import OmegaConf
method = OmegaConf.load(sys.argv[1])
baseline = OmegaConf.load(sys.argv[2])
items = json.load(open(sys.argv[3], encoding="utf-8"))
assert sum(x["suite"] == "scene" for x in items) == 86
assert sum(x["suite"] == "overall_consistency" for x in items) == 93
assert method.memory_control_type == "local_history"
assert method.memory_full_frames == 18 and method.memory_local_frames == 6
assert "memory_max_timestep" not in method
assert baseline.use_memory_dmd is False
print("REMAINING3_AUDIT:scene=86x5=430:overall=93x5=465")
print("CONFIG_AUDIT:method=A18-A6_no_gate_step600:baseline=memory_dmd_false_step600")
PY

generate_suite() {
  local label=$1 config=$2 checkpoint=$3 suite=$4 expected=$5 videos=$6
  mkdir -p "$videos"
  local count
  count=$(video_count "$videos")
  if [[ "$count" -lt "$expected" ]]; then
    cd "$BASE"
    export PYTHONPATH="$BASE:$BASE/long_video"
    export OMP_NUM_THREADS=8
    echo "START_GENERATION:$label:$suite:expected${expected}:8gpu:batch4:all_rank_seed0"
    "$GEN_PY" -m torch.distributed.run --standalone --nproc_per_node=8 \
      tools/vbench_generate.py \
      --config_path "$config" \
      --checkpoint_path "$checkpoint" \
      --manifest "$MANIFEST" \
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
    echo "VIDEO_COUNT_MISMATCH:$label:$suite:expected${expected}:found${count}" >&2
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
  local label=$1 videos=$2 dim=$3 out=$4
  mkdir -p "$out"
  local existing
  existing=$(find "$out" -maxdepth 1 -type f -name '*_eval_results.json' -print -quit)
  if [[ -n "$existing" ]]; then
    echo "SKIP_COMPLETE:$label:$dim:$existing"
    return
  fi
  cd "$VBENCH_ROOT"
  export PYTHONPATH="$VBENCH_ROOT"
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
  export VBENCH_TORCH_THREADS=1
  echo "START_DIMENSION:$label:$dim:8gpu"
  "$EVAL_PY" -m torch.distributed.run --standalone --nproc_per_node=8 evaluate.py \
    --videos_path "$videos" \
    --output_path "$out" \
    --dimension "$dim" \
    --load_ckpt_from_local True \
    --mode vbench_standard
  existing=$(find "$out" -maxdepth 1 -type f -name '*_eval_results.json' -print -quit)
  [[ -n "$existing" ]] || { echo "MISSING_RESULT:$label:$dim" >&2; exit 1; }
  echo "DONE_DIMENSION:$label:$dim:$existing"
}

write_quality7() {
  local label=$1 subject3=$2 tf_results=$3 remaining=$4 out=$5
  mkdir -p "$out"
  "$EVAL_PY" - "$label" "$subject3" "$tf_results" "$remaining" "$out" <<'PY'
import glob, json, os, sys
from pathlib import Path

label, subject3, tf_root, remaining, out = sys.argv[1:]
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
score = sum(normalized[k] * weights[k] for k in bounds) / sum(weights.values())
payload = {
    "label": label,
    "raw": raw,
    "normalized": normalized,
    "weights": weights,
    "quality_score_all7": score,
}
Path(out, "quality_summary_all7.json").write_text(
    json.dumps(payload, indent=2), encoding="utf-8"
)
print(f"QUALITY7:{label}:{score * 100:.8f}")
PY
  touch "$out/COMPLETE"
}

run_model() {
  local label=$1 config=$2 checkpoint=$3 subject3=$4 tf_results=$5 prefix=$6
  local scene_videos="$BASE/outputsv1/${prefix}_scene"
  local overall_videos="$BASE/outputsv1/${prefix}_overall"
  local remaining="$BASE/outputsv1/${prefix}_remaining3_results"
  local quality7="$BASE/outputsv1/${prefix}_quality7_results"

  generate_suite "$label" "$config" "$checkpoint" scene 430 "$scene_videos"
  evaluate_dim "$label" "$scene_videos" background_consistency \
    "$remaining/by_dimension/background_consistency"
  generate_suite "$label" "$config" "$checkpoint" overall_consistency 465 "$overall_videos"
  evaluate_dim "$label" "$overall_videos" aesthetic_quality \
    "$remaining/by_dimension/aesthetic_quality"
  evaluate_dim "$label" "$overall_videos" imaging_quality \
    "$remaining/by_dimension/imaging_quality"
  write_quality7 "$label" "$subject3" "$tf_results" "$remaining" "$quality7"
}

run_model method_step600 "$METHOD_CONFIG" "$METHOD_CHECKPOINT" \
  "$BASE/outputsv1/vbench_results_lrl_nogate_step0600_15s_allrankseed0_subject3" \
  "$METHOD_TF_RESULTS" "vbench_lrl_nogate_step0600_15s_allrankseed0"

run_model baseline_step600 "$BASELINE_CONFIG" "$BASELINE_CHECKPOINT" \
  "$BASE/outputsv1/vbench_results_baseline_step0600_15s_allrankseed0_subject3" \
  "$BASELINE_TF_RESULTS" "vbench_baseline_step0600_15s_allrankseed0"

echo "STEP600_REMAINING3_AND_QUALITY7_PAIR_COMPLETE"
