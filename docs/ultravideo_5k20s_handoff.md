# UltraVideo-Long 5K-20s data handoff

This is the only data path used by the 20-second AR teacher configuration:

```text
dataset/UltraVideo-Long-5K-20s/
```

The locked selection contains 5,000 videos, all at least 20 seconds long:
4,900 train and 100 validation. The old 15.6-second server artifacts and helper
entry points were removed so there is only one active AR-teacher data workflow.

## 0. Requirements

- Run commands from the repository root.
- Set `PYTHON` to the LongVideo environment if `python` is not that interpreter.
- `metadata-20s` and `download-20s` need access to Hugging Face.
- Keep substantial free space for raw videos, approximately 80 GB of latent
  shards, and approximately 80 GB for the final LMDB. The scripts never delete
  raw videos or latent shards automatically.

## 1. Download metadata and videos

```bash
PYTHON=/path/to/python bash tools/ultravideo_long_5k.sh metadata-20s
PYTHON=/path/to/python bash tools/ultravideo_long_5k.sh download-20s
```

The download is resumable. The official data is stored in large ZIP archives,
so all archives are scanned even though only the locked 5,000 clips are kept.
If the GPU server has no internet, run this stage on a connected storage machine
and synchronize the complete `UltraVideo-Long-5K-20s` directory to the server.

## 2. CPU-only preflight

```bash
PYTHON=/path/to/python bash tools/ultravideo_long_5k.sh check-20s
```

Success must report 4,900 assigned training clips and a valid dry run. This
checks every video path and caption without loading the VAE.

## 3. Encode 321 pixel frames to 81 Wan latents

```bash
PYTHON=/path/to/python NPROC_PER_NODE=8 \
  bash tools/ultravideo_long_5k.sh encode-20s
```

Use the actual number of available GPUs for `NPROC_PER_NODE`. The command is
safe to rerun: existing shards are loaded and validated before they are skipped.
Long videos seek directly to their deterministic random crop.

Do not build the LMDB until this file exists:

```text
dataset/UltraVideo-Long-5K-20s/latents_81/_ENCODE_COMPLETE.json
```

## 4. Build and audit the LMDB

```bash
PYTHON=/path/to/python bash tools/ultravideo_long_5k.sh build-20s
PYTHON=/path/to/python FULL_AUDIT=1 \
  bash tools/ultravideo_long_5k.sh audit-20s
```

`build-20s` refuses missing, unexpected, corrupt, non-finite, wrong-shape or
wrong-caption shards. It writes `clean_data_81.building` and only renames it to
`clean_data_81` after all 4,900 entries are committed. If a `.building`
directory remains after an error, inspect and move it aside before retrying.

The final success line starts with:

```text
ULTRAVIDEO_20S_DATA_AUDIT_OK:count=4900
```

## 5. Launch training

The production launcher performs a fast sampled data audit again before any
distributed model process starts. It supports exactly the two reviewed
topologies below. Both use one batch per optimizer update with no gradient
accumulation and learning rate `2e-6`:

| Hardware | Config selected automatically | FSDP | Global batch |
|---|---|---|---:|
| 1 × 8 GPUs | `ar_teacher_ultravideo_long_5k_81f_sparse12_8gpu.yaml` | `full` | 8 |
| 2 × 8 GPUs | `ar_teacher_ultravideo_long_5k_81f_sparse12.yaml` | `hybrid_full` | 16 |

Single-node 8-GPU launch:

```bash
PYTHON=/path/to/python bash tools/run_ar_teacher_sparse12_20s.sh
```

Two-node 16-GPU launch (run on each node with `NODE_RANK=0` or `1`):

```bash
NNODES=2 NODE_RANK=... MASTER_ADDR=... \
  PYTHON=/path/to/python \
  bash tools/run_ar_teacher_sparse12_20s.sh
```

Set `DRY_RUN=1` on either command to validate topology, config, global batch,
absence of gradient accumulation, sharding, and the sparse attention contract
without requiring the dataset or checkpoint. Unsupported layouts fail before
training starts.

Never manually copy an LMDB from another run into this root. The encode-plan
digest is carried into LMDB metadata specifically to detect that mistake.
