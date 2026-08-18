# UltraVideo-Long 5K data handoff

This repository does **not** redistribute UltraVideo-Long MP4 files. The
official license permits research use but explicitly prohibits redistributing
the raw videos, transcripts or thumbnails. Keep raw videos on the training
server and share only the manifest, scripts, and (if approved by the dataset
owner) derived latent features.

## Reproducible subset

The locked subset is selected from `long.csv` with:

- source: `APRIL-AIGC/UltraVideo-Long`
- resolution archive: 960p
- count: 5,000 (4,900 train + 100 validation)
- seed: 42
- minimum duration: 15.6 seconds
- temporal crop: deterministic per `seed:clip_id`
- output: 249 frames at 16fps, center-cropped to 480x832

## 1. Download metadata and lock the manifest

```bash
python tools/download_ultravideo_long_5k.py \
  --output-dir dataset/UltraVideo-Long-5K \
  --selection-ids data_manifests/ultravideo_long_5k/selection_ids.csv \
  --count 5000 --seed 42 --resolution 960 \
  --min-duration 15.6 --val-count 100 \
  --metadata-only
```

Only `selection_ids.csv` (clip ID, index and split) and the selection checksum
are suitable for the code repository. The full CSV/JSONL and captions are
regenerated from the official metadata on the training server; do not publish
them independently. Never commit `videos/`.

## 2. Resumable raw-video download

Run the same command without `--metadata-only`:

```bash
python tools/download_ultravideo_long_5k.py \
  --output-dir dataset/UltraVideo-Long-5K \
  --selection-ids data_manifests/ultravideo_long_5k/selection_ids.csv \
  --count 5000 --seed 42 --resolution 960 \
  --min-duration 15.6 --val-count 100
```

The downloader scans one official ZIP at a time, resumes a `.part` file with
HTTP Range, verifies extracted entry size/ZIP CRC, saves an archive marker, and
deletes the scanned ZIP. Interrupting it is safe; rerun the identical command.
It must scan all eight 960p archives (about 293GB network traffic) to select an
unbiased subset, but retains only the chosen clips.

## 3. Validate preprocessing on a few clips

This does not load the VAE and is safe on a login node:

```bash
python tools/prepare_ultravideo_long_latents.py \
  --manifest dataset/UltraVideo-Long-5K/manifests/train.csv \
  --videos-dir dataset/UltraVideo-Long-5K/videos \
  --output-dir dataset/UltraVideo-Long-5K/latents_63 \
  --num-frames 249 --limit 8 --dry-run
```

## 4. Encode long clips on 8 GPUs

```bash
torchrun --standalone --nproc_per_node=8 \
  tools/prepare_ultravideo_long_latents.py \
  --manifest dataset/UltraVideo-Long-5K/manifests/train.csv \
  --videos-dir dataset/UltraVideo-Long-5K/videos \
  --output-dir dataset/UltraVideo-Long-5K/latents_63 \
  --num-frames 249 --fps 16 --height 480 --width 832 --seed 42
```

Each finished clip is saved atomically as one float16 latent shard. Rerunning
the command skips completed shards. `249` pixel frames become `63` Wan latent
frames. For a strict reproduction of the original 5-second AR-teacher domain,
use `--num-frames 81` and a separate output directory.

## 5. Build the training LMDB

```bash
python tools/build_ultravideo_long_lmdb.py \
  --latents-dir dataset/UltraVideo-Long-5K/latents_63 \
  --output dataset/UltraVideo-Long-5K/clean_data_63
```

The existing `LatentLMDBDataset` can read this schema. A true long-history AR
teacher must also increase the wrapper sequence capacity and train through the
63-latent teacher-forcing mask. This repository provides both a one-GPU smoke
configuration and the intended two-node/16-GPU 2K-step configuration.

## 6. One-GPU end-to-end smoke test

The smoke path keeps all 63 latent frames but uses 240x416 pixels (30x52
latents) so it tests long temporal indexing without requiring production
activation memory:

```bash
bash tools/ultravideo_long_5k.sh debug-download
bash tools/ultravideo_long_5k.sh debug-encode
bash tools/ultravideo_long_5k.sh debug-build
bash tools/run_ar_teacher_ultravideo_long_smoke_1gpu.sh
```

Success requires two optimizer steps, two nonempty checkpoints, a clean
`max_steps=2` exit, and no OOM/NaN/traceback. This is a code-path test, not a
quality experiment.

## 7. Production 2K-step fine-tune

After both machines have identical code, base checkpoints and LMDB paths, run
the multinode script on both machines with the same `MASTER_ADDR`:

```bash
# machine 0
NODE_RANK=0 MASTER_ADDR=<machine-0-ip> \
  bash tools/run_ar_teacher_ultravideo_long_2k_multinode.sh

# machine 1
NODE_RANK=1 MASTER_ADDR=<machine-0-ip> \
  bash tools/run_ar_teacher_ultravideo_long_2k_multinode.sh
```

The production configuration is
`configs/ar_teacher_ultravideo_long_5k_63f_2k.yaml`: seed 42, 63 latents,
full causal history, batch 1 per GPU and effective global batch 16. It is
open-ended and saves every 100 steps; 2K is only the planned observation point,
and the operator decides when to stop. The trainer rejects a world-size/global-
batch mismatch rather than silently changing the optimization protocol.

Both launchers default `WAN_FLEX_COMPILE_MODE` to
`max-autotune-no-cudagraphs`. This is required on the tested RTX PRO 6000
Blackwell host: the fixed default Triton schedule exceeded per-block shared
resources, while autotuning selected valid, mathematically equivalent forward
and backward FlexAttention kernels. It may be overridden explicitly on a
known-good older cluster.

## Storage and publishing boundary

- GitHub: code, configs, manifests, documentation, small checksums.
- Training server: raw MP4s, `.pt` latent shards, LMDB, checkpoints and logs.
- Optional private artifact storage: derived latents only after confirming the
  upstream license interpretation with the dataset owner.
