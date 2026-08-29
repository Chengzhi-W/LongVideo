#!/usr/bin/env python3
"""Audit a completed 20-second UltraVideo latent dataset and its LMDB."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.build_ultravideo_long_lmdb import load_encode_contract


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the manifest, encode contract, shards and clean LMDB."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--videos-dir", type=Path)
    parser.add_argument("--latents-dir", type=Path, required=True)
    parser.add_argument("--lmdb", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=0)
    parser.add_argument("--min-duration", type=float, default=20.0)
    parser.add_argument(
        "--latent-shape", type=int, nargs=4, default=(81, 16, 60, 104)
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Read and compare every LMDB item instead of first/middle/last.",
    )
    return parser.parse_args()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_eligible_rows(path: Path, min_duration: float) -> tuple[list[dict], int]:
    with path.resolve().open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"clip_id", "local_path", "text", "duration"}
    fields = set(rows[0]) if rows else set()
    missing = required - fields
    if missing:
        raise RuntimeError(f"Manifest lacks required columns: {sorted(missing)}")

    eligible = []
    skipped = 0
    seen = set()
    for row in rows:
        clip_id = row["clip_id"].strip()
        if not clip_id or clip_id in seen:
            raise RuntimeError(f"Empty or duplicate clip_id: {clip_id!r}")
        seen.add(clip_id)
        if not row["text"].strip():
            raise RuntimeError(f"Empty caption for {clip_id}")
        try:
            duration = float(row["duration"])
        except ValueError as exc:
            raise RuntimeError(f"Invalid duration for {clip_id}") from exc
        if duration + 1e-3 < min_duration:
            skipped += 1
            continue
        eligible.append(row)
    return eligible, skipped


def audit_plan(args: argparse.Namespace, rows: list[dict]) -> dict:
    plan = load_encode_contract(args.latents_dir.resolve(), True)
    if plan is None:
        raise AssertionError("require_complete=True must return an encode plan")
    manifest_digest = hashlib.sha256(args.manifest.resolve().read_bytes()).hexdigest()
    if plan.get("manifest_sha256") != manifest_digest:
        raise RuntimeError("Manifest content no longer matches the encode plan")
    expected_shape = tuple(args.latent_shape)
    if tuple(plan["latent_shape"]) != expected_shape:
        raise RuntimeError(
            f"Encode plan latent shape {plan['latent_shape']} != {expected_shape}"
        )
    if plan["count"] != len(rows):
        raise RuntimeError(
            f"Encode plan count {plan['count']} != eligible manifest rows {len(rows)}"
        )
    if args.expected_count and len(rows) != args.expected_count:
        raise RuntimeError(
            f"Eligible manifest count {len(rows)} != --expected-count {args.expected_count}"
        )

    for index, (sample, row) in enumerate(zip(plan["samples"], rows)):
        expected = {
            "clip_id": row["clip_id"],
            "prompt_sha256": sha256_text(row["text"].strip()),
            "source_video": Path(row["local_path"]).name,
        }
        actual = {key: sample.get(key) for key in expected}
        if actual != expected:
            raise RuntimeError(
                f"Encode plan sample {index} disagrees with manifest: "
                f"actual={actual}, expected={expected}"
            )
    return plan


def audit_videos(videos_dir: Path | None, rows: list[dict]) -> None:
    if videos_dir is None:
        return
    root = videos_dir.resolve()
    missing = [
        row["clip_id"]
        for row in rows
        if not (root / Path(row["local_path"]).name).is_file()
    ]
    if missing:
        raise RuntimeError(
            f"Missing {len(missing)} source videos, first entries: {missing[:8]}"
        )


def audit_lmdb(args: argparse.Namespace, plan: dict) -> None:
    import lmdb
    import numpy as np

    path = args.lmdb.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"LMDB directory does not exist: {path}")
    env = lmdb.open(
        str(path), readonly=True, lock=False, readahead=False, meminit=False
    )
    try:
        with env.begin() as txn:
            shape_raw = txn.get(b"latents_shape")
            metadata_raw = txn.get(b"dataset_metadata")
            if shape_raw is None or metadata_raw is None:
                raise RuntimeError("LMDB is missing shape or dataset metadata")
            shape = tuple(map(int, shape_raw.decode().split()))
            metadata = json.loads(metadata_raw)
        expected_shape = (plan["count"], *tuple(args.latent_shape))
        if shape != expected_shape:
            raise RuntimeError(f"LMDB shape {shape} != {expected_shape}")
        if metadata.get("count") != plan["count"]:
            raise RuntimeError("LMDB metadata count does not match the encode plan")
        if metadata.get("encode_plan_sha256") != plan["plan_sha256"]:
            raise RuntimeError("LMDB was not built from the current encode plan")

        if args.full:
            indices = range(plan["count"])
        else:
            indices = sorted({0, plan["count"] // 2, plan["count"] - 1})
        latent_shape = tuple(args.latent_shape)
        for index in indices:
            sample = plan["samples"][index]
            with env.begin() as txn:
                latent_raw = txn.get(f"latents_{index}_data".encode())
                prompt_raw = txn.get(f"prompts_{index}_data".encode())
                clip_raw = txn.get(f"clip_ids_{index}_data".encode())
            if latent_raw is None or prompt_raw is None or clip_raw is None:
                raise RuntimeError(f"LMDB item {index} is incomplete")
            latent = np.frombuffer(latent_raw, dtype=np.float16)
            if latent.size != int(np.prod(latent_shape)) or not np.isfinite(latent).all():
                raise RuntimeError(f"LMDB latent {index} has invalid size or values")
            prompt = prompt_raw.decode("utf-8")
            clip_id = clip_raw.decode("utf-8")
            if sha256_text(prompt) != sample["prompt_sha256"]:
                raise RuntimeError(f"LMDB prompt mismatch at index {index}")
            if clip_id != sample["clip_id"]:
                raise RuntimeError(f"LMDB clip_id mismatch at index {index}")
    finally:
        env.close()


def main() -> int:
    args = parse_args()
    rows, skipped = load_eligible_rows(args.manifest, args.min_duration)
    if not rows:
        raise RuntimeError("No eligible manifest rows")
    audit_videos(args.videos_dir, rows)
    plan = audit_plan(args, rows)
    audit_lmdb(args, plan)
    mode = "full" if args.full else "sampled"
    print(
        "ULTRAVIDEO_20S_DATA_AUDIT_OK:"
        f"count={len(rows)}:short_rows_skipped={skipped}:"
        f"latent_shape={tuple(args.latent_shape)}:lmdb_check={mode}:"
        f"plan_sha256={plan['plan_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
