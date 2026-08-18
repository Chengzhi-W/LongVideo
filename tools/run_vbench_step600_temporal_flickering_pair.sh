#!/usr/bin/env bash
set -euo pipefail

BASE=${BASE:-/root/autodl-tmp/Causal-Forcing}
VBENCH_ROOT=${VBENCH_ROOT:-/root/autodl-tmp/VBench}
GEN_PY=${GEN_PY:-/root/autodl-tmp/envs/longlive/bin/python}
EVAL_PY=${EVAL_PY:-/root/autodl-tmp/envs/vbench/bin/python}

METHOD_CONFIG="$BASE/configs/causal_forcing_dmd_chunkwise_long_vs_local_joint_nogate_step1500.yaml"
METHOD_CHECKPOINT="$BASE/logsv1/repro8x_long_vs_local_joint_nogate_step1500_seed42/checkpoint_model_000600/model.pt"
BASELINE_CONFIG="$BASE/configs/causal_forcing_dmd_chunkwise_senior_baseline_step1500.yaml"
BASELINE_CHECKPOINT="$BASE/logsv1/repro8x_causal_forcing_dmd_seed42_baseline_step1500_chunkwise/checkpoint_model_000600/model.pt"
MANIFEST="$BASE/prompts/vbench_temporal_flickering_seed42.json"
METHOD_VIDEOS="$BASE/outputsv1/vbench_lrl_nogate_step0600_15s_allrankseed0_temporal_flickering"
BASELINE_VIDEOS="$BASE/outputsv1/vbench_baseline_step0600_15s_allrankseed0_temporal_flickering"
METHOD_RESULTS="$BASE/outputsv1/vbench_results_lrl_nogate_step0600_15s_allrankseed0_temporal_flickering"
BASELINE_RESULTS="$BASE/outputsv1/vbench_results_baseline_step0600_15s_allrankseed0_temporal_flickering"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VBENCH_TORCH_THREADS=1

video_count() {
  find "$1" -maxdepth 1 -type f -name '*.mp4' -size +0c 2>/dev/null | wc -l
}

for path in "$METHOD_CONFIG" "$METHOD_CHECKPOINT" "$BASELINE_CONFIG" \
  "$BASELINE_CHECKPOINT" "$MANIFEST"; do
  [[ -s "$path" ]] || { echo "MISSING_REQUIRED_FILE:$path" >&2; exit 1; }
done

"$GEN_PY" - "$METHOD_CONFIG" "$BASELINE_CONFIG" "$MANIFEST" <<'PY'
import json, sys
from omegaconf import OmegaConf

method = OmegaConf.load(sys.argv[1])
baseline = OmegaConf.load(sys.argv[2])
items = json.load(open(sys.argv[3], encoding="utf-8"))
assert len(items) == 75, len(items)
assert all(x["suite"] == "temporal_flickering" for x in items)
assert len({x["prompt"] for x in items}) == 75
assert method.memory_control_type == "local_history"
assert method.memory_full_frames == 18 and method.memory_local_frames == 6
assert "memory_max_timestep" not in method
assert baseline.use_memory_dmd is False
print("TF_AUDIT:official_prompts=75:samples=5:videos_per_model=375")
print("CONFIG_AUDIT:method=A18-A6_no_gate_step600:baseline=memory_dmd_false_step600")
PY

generate_tf() {
  local label=$1
  local config=$2
  local checkpoint=$3
  local videos=$4
  mkdir -p "$videos"
  local count
  count=$(video_count "$videos")
  if [[ "$count" -lt 375 ]]; then
    cd "$BASE"
    export PYTHONPATH="$BASE:$BASE/long_video"
    export OMP_NUM_THREADS=8
    echo "START_TF_GENERATION:$label:8gpu:batch4:all_rank_seed0:63_latent:375"
    "$GEN_PY" -m torch.distributed.run --standalone --nproc_per_node=8 \
      tools/vbench_generate.py \
      --config_path "$config" \
      --checkpoint_path "$checkpoint" \
      --manifest "$MANIFEST" \
      --output_folder "$videos" \
      --suite temporal_flickering \
      --num_output_frames 63 \
      --samples_per_prompt 5 \
      --batch_size 4 \
      --base_seed 0 \
      --seed_mode prompt_batch_stream \
      --use_ema
  fi

  count=$(video_count "$videos")
  [[ "$count" -eq 375 ]] || { echo "EXPECTED_375_VIDEOS:$label:found$count" >&2; exit 1; }
  "$GEN_PY" - "$videos" <<'PY'
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
assert len(paths) == 375, len(paths)
assert not bad, bad[:20]
print("TF_VIDEO_AUDIT:count=375:frames=249:fps=16:duration=15.5625s")
PY
}

run_tf() {
  local label=$1
  local videos=$2
  local results=$3
  mkdir -p "$results"
  local existing
  existing=$(find "$results" -maxdepth 1 -type f -name '*_eval_results.json' -print -quit)
  if [[ -n "$existing" ]]; then
    echo "SKIP_COMPLETE:$label:$existing"
    return
  fi

  cd "$VBENCH_ROOT"
  export PYTHONPATH="$VBENCH_ROOT"
  echo "START_TEMPORAL_FLICKERING:$label:8gpu:375_official_tf_videos"
  "$EVAL_PY" -m torch.distributed.run --standalone --nproc_per_node=8 evaluate.py \
    --videos_path "$videos" \
    --output_path "$results" \
    --dimension temporal_flickering \
    --load_ckpt_from_local True \
    --mode vbench_standard

  existing=$(find "$results" -maxdepth 1 -type f -name '*_eval_results.json' -print -quit)
  [[ -n "$existing" ]] || { echo "MISSING_TEMPORAL_FLICKERING_RESULT:$label" >&2; exit 1; }
  echo "DONE_TEMPORAL_FLICKERING:$label:$existing"
}

generate_tf method_step600 "$METHOD_CONFIG" "$METHOD_CHECKPOINT" "$METHOD_VIDEOS"
run_tf method_step600 "$METHOD_VIDEOS" "$METHOD_RESULTS"
generate_tf baseline_step600 "$BASELINE_CONFIG" "$BASELINE_CHECKPOINT" "$BASELINE_VIDEOS"
run_tf baseline_step600 "$BASELINE_VIDEOS" "$BASELINE_RESULTS"

touch "$METHOD_RESULTS/COMPLETE" "$BASELINE_RESULTS/COMPLETE"
echo "STEP600_TEMPORAL_FLICKERING_PAIR_COMPLETE"
