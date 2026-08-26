# 20-second AR teacher fine-tuning

This stage fine-tunes `checkpoints/chunkwise/ar_diffusion.pt` on 81-latent
(20-second) clips with normal flow matching. It does not use frequency
decomposition, KV dropout, or DMD.

## Attention contract

Each three-latent current chunk has a strict 21-frame visible budget:

```text
sink3 + slow12 + fast recent3 + noisy current3 = 21
```

Slow memory keeps the first latent frame of every second historical chunk. A
frame still present in the fast window is excluded from slow memory. For the
last 20-second target (`78..80`), the clean history is:

```text
sink: 0,1,2
slow: 3,9,15,21,27,33,39,45,51,57,63,69
fast: 75,76,77
current noisy: 78,79,80
```

The training mask and inference KV cache use the same fixed schedule and the
same absolute RoPE positions.

## Sparse-target teacher forcing

The clean 81-frame history is retained. Each optimizer step packs 12 of the 27
complete noisy current chunks, so the transformer sequence has 117 frames
instead of 162 (`clean81 + noisy36`, a 27.8% reduction in sequence-dependent
activations). The nine-pattern bank is exactly balanced: every chunk appears in
four patterns, and each rank randomly chooses a pattern every step.

Twelve targets is intentionally not a tiny sample. With global batch 16 it
provides 192 supervised chunks per optimizer step while bounding the number of
FlexAttention masks compiled by each rank.

## Data and launch

Prepare 20-second latents from 321 pixel frames at 16 fps:

```bash
python -m torch.distributed.run --nproc_per_node=8 \
  tools/prepare_ultravideo_long_latents.py \
  --manifest DATASET_MANIFEST.csv \
  --videos-dir DATASET_VIDEOS \
  --output-dir dataset/UltraVideo-Long-5K/latents_81 \
  --num-frames 321 --fps 16 --height 480 --width 832 \
  --skip-too-short
```

Build the LMDB:

```bash
python tools/build_ultravideo_long_lmdb.py \
  --latents-dir dataset/UltraVideo-Long-5K/latents_81 \
  --output dataset/UltraVideo-Long-5K/clean_data_81
```

The same two operations are available as `encode-20s` and `build-20s` in
`tools/ultravideo_long_5k.sh`. Before training, run:

```bash
python tools/audit_sparse_teacher_forcing_20s.py \
  --config configs/ar_teacher_ultravideo_long_5k_81f_sparse12.yaml
```

Use `tools/run_ar_teacher_sparse12_20s_smoke_1gpu.sh` for the two-step GPU
smoke test and `tools/run_ar_teacher_sparse12_20s_multinode.sh` for production.
