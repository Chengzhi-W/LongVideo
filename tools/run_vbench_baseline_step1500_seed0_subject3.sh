#!/usr/bin/env bash
set -euo pipefail

BASE=${BASE:-/root/autodl-tmp/Causal-Forcing}
VBENCH_ROOT=${VBENCH_ROOT:-/root/autodl-tmp/VBench}
GEN_PY=${GEN_PY:-/root/autodl-tmp/envs/longlive/bin/python}
EVAL_PY=${EVAL_PY:-/root/autodl-tmp/envs/vbench/bin/python}
STEP=${STEP:-1500}

[[ "$STEP" =~ ^[0-9]{4}$ ]] || { echo "INVALID_STEP:$STEP" >&2; exit 1; }

CONFIG="$BASE/configs/causal_forcing_dmd_chunkwise_senior_baseline_step1500.yaml"
CHECKPOINT="$BASE/logsv1/repro8x_causal_forcing_dmd_seed42_baseline_step1500_chunkwise/checkpoint_model_00${STEP}/model.pt"
MANIFEST="$BASE/prompts/vbench_quality_no_tf_seed42.json"
VIDEOS="$BASE/outputsv1/vbench_baseline_step${STEP}_15s_allrankseed0_subject"
RESULTS="$BASE/outputsv1/vbench_results_baseline_step${STEP}_15s_allrankseed0_subject3"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

# Subject, Motion, and Dynamic use the exact same official 72 prompts.
# Put Dynamic before the much slower Motion evaluation for faster feedback.
DIMS=(subject_consistency dynamic_degree motion_smoothness)

video_count() {
  find "$VIDEOS" -maxdepth 1 -type f -name '*.mp4' -size +0c 2>/dev/null | wc -l
}

[[ -s "$CONFIG" ]] || { echo "MISSING_CONFIG:$CONFIG" >&2; exit 1; }
[[ -s "$CHECKPOINT" ]] || { echo "MISSING_CHECKPOINT:$CHECKPOINT" >&2; exit 1; }
[[ -s "$MANIFEST" ]] || { echo "MISSING_MANIFEST:$MANIFEST" >&2; exit 1; }
mkdir -p "$VIDEOS" "$RESULTS/by_dimension"

"$GEN_PY" - "$CONFIG" "$MANIFEST" "$STEP" <<'PY'
import json, sys
from omegaconf import OmegaConf

c = OmegaConf.load(sys.argv[1])
items = json.load(open(sys.argv[2], encoding="utf-8"))
subject = [item for item in items if item["suite"] == "subject_consistency"]
assert len(subject) == 72, len(subject)
assert len({item["prompt"] for item in subject}) == 72
assert c.seed == 42
assert c.use_memory_dmd is False
print(f"CONFIG_AUDIT:baseline:memory_dmd_false:step{sys.argv[3]}_EMA")
print("MANIFEST_AUDIT:subject_prompts=72:samples=5:videos=360:shared_by=subject,dynamic,motion")
PY

count=$(video_count)
if [[ "$count" -lt 360 ]]; then
  cd "$BASE"
  export PYTHONPATH="$BASE:$BASE/long_video"
  echo "START_GENERATION:baseline:step${STEP}:EMA:8gpu:batch4:all_rank_seed0:63_latent:360"
  "$GEN_PY" -m torch.distributed.run --standalone --nproc_per_node=8 \
    tools/vbench_generate.py \
    --config_path "$CONFIG" \
    --checkpoint_path "$CHECKPOINT" \
    --manifest "$MANIFEST" \
    --output_folder "$VIDEOS" \
    --suite subject_consistency \
    --num_output_frames 63 \
    --samples_per_prompt 5 \
    --batch_size 4 \
    --base_seed 0 \
    --seed_mode prompt_batch_stream \
    --use_ema
fi

count=$(video_count)
[[ "$count" -eq 360 ]] || { echo "EXPECTED_360_VALID_VIDEOS:found${count}" >&2; exit 1; }

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
assert len(paths) == 360, len(paths)
assert not bad, bad[:20]
print("VIDEO_AUDIT:count=360:frames=249:fps=16:duration=15.5625s")
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
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
  export VBENCH_TORCH_THREADS=1
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
[[ "${#result_files[@]}" -eq 3 ]] || {
  echo "EXPECTED_3_RESULTS:found${#result_files[@]}" >&2
  exit 1
}
"$EVAL_PY" "$BASE/tools/merge_vbench_dimension_results.py" \
  --output "$RESULTS/combined_eval_results.json" \
  --expected-dimensions subject_consistency,dynamic_degree,motion_smoothness \
  "${result_files[@]}"
touch "$RESULTS/COMPLETE"
echo "BASELINE_STEP${STEP}_SUBJECT_DYNAMIC_MOTION_COMPLETE:$RESULTS"
