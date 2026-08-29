#!/usr/bin/env python3
"""Build the clean-latent LMDB consumed by LatentLMDBDataset.

Input shards come from ``prepare_ultravideo_long_latents.py``.  The operation
is atomic at the directory level: it writes a sibling ``.building`` LMDB and
renames it only after all shapes and metadata have been committed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


ENCODE_PLAN_NAME = "_ENCODE_PLAN.json"
ENCODE_COMPLETE_NAME = "_ENCODE_COMPLETE.json"
ENCODE_SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pack UltraVideo latent shards into LMDB")
    parser.add_argument("--latents-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--map-size-gb", type=int, default=1024)
    parser.add_argument("--commit-every", type=int, default=32)
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help=(
            "Require and validate the encoder's completion contract. Use this "
            "for all 20-second production and smoke datasets."
        ),
    )
    return parser.parse_args()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_encode_contract(latents_dir: Path, require_complete: bool) -> dict | None:
    plan_path = latents_dir / ENCODE_PLAN_NAME
    complete_path = latents_dir / ENCODE_COMPLETE_NAME
    if not plan_path.exists() or not complete_path.exists():
        if require_complete:
            raise RuntimeError(
                "Encoded dataset is not marked complete. Rerun the exact encoding "
                f"command until both {plan_path.name} and {complete_path.name} exist."
            )
        return None

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    if plan.get("schema_version") != ENCODE_SCHEMA_VERSION:
        raise RuntimeError(f"Unsupported encode plan schema: {plan_path}")
    if complete.get("schema_version") != ENCODE_SCHEMA_VERSION or not complete.get(
        "complete"
    ):
        raise RuntimeError(f"Invalid encode completion marker: {complete_path}")

    plan_for_digest = dict(plan)
    recorded_digest = plan_for_digest.pop("plan_sha256", None)
    canonical = json.dumps(
        plan_for_digest, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    actual_digest = hashlib.sha256(canonical).hexdigest()
    if recorded_digest != actual_digest:
        raise RuntimeError(
            f"Encode plan digest mismatch: {recorded_digest} != {actual_digest}"
        )
    if complete.get("plan_sha256") != actual_digest:
        raise RuntimeError("Completion marker does not belong to the current encode plan")
    if complete.get("count") != plan.get("count"):
        raise RuntimeError("Completion marker count does not match the encode plan")
    if complete.get("latent_shape") != plan.get("latent_shape"):
        raise RuntimeError("Completion marker latent shape does not match the encode plan")

    expected = [sample["filename"] for sample in plan["samples"]]
    if len(expected) != plan["count"] or len(set(expected)) != len(expected):
        raise RuntimeError("Encode plan contains an invalid or duplicate sample list")
    actual = {path.name for path in latents_dir.glob("*.pt")}
    expected_set = set(expected)
    missing = sorted(expected_set - actual)
    unexpected = sorted(actual - expected_set)
    if missing or unexpected:
        raise RuntimeError(
            "Refusing to build a partial/contaminated LMDB: "
            f"missing={missing[:8]} ({len(missing)} total), "
            f"unexpected={unexpected[:8]} ({len(unexpected)} total)"
        )
    return plan


def validate_shard(torch, path: Path, item: dict, sample: dict, latent_shape: tuple):
    if item.get("clip_id") != sample["clip_id"]:
        raise RuntimeError(
            f"clip_id mismatch in {path}: {item.get('clip_id')} != {sample['clip_id']}"
        )
    prompt = str(item.get("prompt") or "")
    if not prompt or sha256_text(prompt) != sample["prompt_sha256"]:
        raise RuntimeError(f"Prompt is empty or mismatched in {path}")
    if item.get("source_video") != sample["source_video"]:
        raise RuntimeError(f"source_video mismatch in {path}")
    latent = item.get("latent")
    if not torch.is_tensor(latent) or tuple(latent.shape) != latent_shape:
        raise RuntimeError(
            f"Latent shape mismatch in {path}: "
            f"{getattr(latent, 'shape', None)} != {latent_shape}"
        )
    if latent.dtype != torch.float16 or not bool(torch.isfinite(latent).all()):
        raise RuntimeError(f"Latent must be finite float16 in {path}")
    return latent.contiguous().numpy(), prompt


def main() -> int:
    args = parse_args()
    if args.map_size_gb <= 0 or args.commit_every <= 0:
        raise ValueError("--map-size-gb and --commit-every must be positive")
    import lmdb
    import torch

    latents_dir = args.latents_dir.resolve()
    contract = load_encode_contract(latents_dir, args.require_complete)
    if contract is None:
        files = sorted(latents_dir.glob("*.pt"))
        samples = None
        expected_shape = None
    else:
        files = [latents_dir / sample["filename"] for sample in contract["samples"]]
        samples = contract["samples"]
        expected_shape = tuple(contract["latent_shape"])
    if not files:
        raise RuntimeError(f"No .pt shards found under {args.latents_dir}")
    destination = args.output.resolve()
    building = destination.with_name(destination.name + ".building")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing LMDB: {destination}")
    if building.exists():
        raise FileExistsError(
            f"Incomplete build already exists: {building}. Inspect/remove it before retrying."
        )
    building.mkdir(parents=True)
    env = lmdb.open(
        str(building),
        map_size=args.map_size_gb * 1024**3,
        subdir=True,
        map_async=True,
        writemap=False,
    )
    latent_shape = None
    txn = env.begin(write=True)
    try:
        for index, path in enumerate(files):
            item = torch.load(path, map_location="cpu", weights_only=False)
            if samples is None:
                latent_tensor = item["latent"]
                if not torch.is_tensor(latent_tensor):
                    raise RuntimeError(f"Missing latent tensor in {path}")
                latent = latent_tensor.to(dtype=torch.float16).contiguous().numpy()
                prompt = str(item["prompt"])
            else:
                latent, prompt = validate_shard(
                    torch, path, item, samples[index], expected_shape
                )
            if latent_shape is None:
                latent_shape = tuple(latent.shape)
            elif tuple(latent.shape) != latent_shape:
                raise RuntimeError(
                    f"Mixed latent shapes: {path} has {tuple(latent.shape)}, "
                    f"expected {latent_shape}"
                )
            txn.put(f"latents_{index}_data".encode(), latent.tobytes())
            txn.put(f"prompts_{index}_data".encode(), prompt.encode("utf-8"))
            txn.put(f"clip_ids_{index}_data".encode(), str(item["clip_id"]).encode("utf-8"))
            if (index + 1) % args.commit_every == 0:
                txn.commit()
                txn = env.begin(write=True)
                print(f"[pack] {index + 1}/{len(files)}")
        assert latent_shape is not None
        txn.put(
            b"latents_shape",
            " ".join(map(str, (len(files), *latent_shape))).encode("utf-8"),
        )
        txn.put(b"prompts_shape", f"{len(files)}".encode("utf-8"))
        txn.put(b"clip_ids_shape", f"{len(files)}".encode("utf-8"))
        txn.put(
            b"dataset_metadata",
            json.dumps(
                {
                    "count": len(files),
                    "latent_shape": latent_shape,
                    "source": "APRIL-AIGC/UltraVideo-Long",
                    "format": "float16 Wan VAE latents",
                    "encode_plan_sha256": (
                        contract["plan_sha256"] if contract is not None else None
                    ),
                    "preprocess": contract.get("preprocess") if contract else None,
                },
                sort_keys=True,
            ).encode("utf-8"),
        )
        txn.commit()
        txn = None
        env.sync()
        env.close()
        os.replace(building, destination)
    except Exception:
        if txn is not None:
            txn.abort()
        env.close()
        raise
    print(f"[COMPLETE] {len(files)} clips -> {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
