#!/usr/bin/env python3
"""Convert an UltraVideo-Long manifest into resumable Wan VAE latent shards.

Run with ``torchrun`` to split clips deterministically across GPUs.  Every
completed clip is an independent atomic ``.pt`` shard, so an interrupted run
can be restarted without touching finished work.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ENCODE_PLAN_NAME = "_ENCODE_PLAN.json"
ENCODE_COMPLETE_NAME = "_ENCODE_COMPLETE.json"
ENCODE_SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode, crop and VAE-encode UltraVideo-Long clips."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--num-frames",
        type=int,
        required=True,
        help=(
            "Pixel frames per clip: use 81 for 21 latents, 249 for 63 latents, "
            "or 321 for 81 latents (20 seconds at 16 fps)."
        ),
    )
    parser.add_argument("--fps", type=float, default=16.0)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split", choices=("train", "validation", "all"), default="train"
    )
    parser.add_argument(
        "--caption-field",
        default="text",
        help="Manifest caption column (default: text).",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Process at most N clips (0 means all)."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths and print the plan without importing GPU models.",
    )
    parser.add_argument(
        "--skip-too-short",
        action="store_true",
        help=(
            "Skip manifest rows whose duration column is shorter than the "
            "requested temporal window instead of failing during decode."
        ),
    )
    return parser.parse_args()


def stable_fraction(seed: int, clip_id: str) -> float:
    payload = f"{seed}:{clip_id}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value / float(2**64 - 1)


def load_manifest(path: Path, split: str) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"clip_id", "local_path"}
    missing = required.difference(rows[0] if rows else {})
    if missing:
        raise RuntimeError(f"Manifest lacks required columns: {sorted(missing)}")
    if split != "all":
        rows = [row for row in rows if row.get("split") == split]
    return rows


def resolve_video(videos_dir: Path, row: dict[str, str]) -> Path:
    candidates = [
        videos_dir / Path(row["local_path"]).name,
        videos_dir / row["clip_id"],
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Video not found for {row['clip_id']}: {candidates}")


def output_name(index: int, clip_id: str) -> str:
    safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in clip_id)
    return f"{index:06d}_{safe}.pt"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def atomic_json_write(destination: Path, payload: dict) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def build_encode_plan(args: argparse.Namespace, rows: list[dict[str, str]]) -> dict:
    samples = []
    for index, row in enumerate(rows):
        prompt = row[args.caption_field].strip()
        samples.append(
            {
                "clip_id": row["clip_id"],
                "duration_seconds": row.get("duration", "").strip(),
                "filename": output_name(index, row["clip_id"]),
                "prompt_sha256": sha256_text(prompt),
                "source_video": Path(row["local_path"]).name,
            }
        )

    payload = {
        "schema_version": ENCODE_SCHEMA_VERSION,
        "source": "APRIL-AIGC/UltraVideo-Long",
        "manifest_sha256": hashlib.sha256(args.manifest.resolve().read_bytes()).hexdigest(),
        "split": args.split,
        "caption_field": args.caption_field,
        "count": len(samples),
        "preprocess": {
            "seed": args.seed,
            "pixel_frames": args.num_frames,
            "latent_frames": (args.num_frames - 1) // 4 + 1,
            "fps": args.fps,
            "height": args.height,
            "width": args.width,
        },
        "latent_shape": [
            (args.num_frames - 1) // 4 + 1,
            16,
            args.height // 8,
            args.width // 8,
        ],
        "samples": samples,
    }
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    payload["plan_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def ensure_encode_plan(output_dir: Path, plan: dict) -> None:
    destination = output_dir / ENCODE_PLAN_NAME
    if destination.exists():
        current = json.loads(destination.read_text(encoding="utf-8"))
        if current != plan:
            raise RuntimeError(
                f"Encoding settings do not match existing {destination}. "
                "Use the original arguments or a new --output-dir."
            )
        return
    atomic_json_write(destination, plan)


def probe_duration_seconds(container, stream) -> float:
    if stream.duration is not None and stream.time_base is not None:
        return float(stream.duration * stream.time_base)
    if container.duration is not None:
        return float(container.duration) / 1_000_000.0
    if stream.frames and stream.average_rate:
        return float(stream.frames / stream.average_rate)
    raise RuntimeError("Video duration is unavailable")


def resize_center_crop(rgb, target_height: int, target_width: int):
    import cv2

    source_height, source_width = rgb.shape[:2]
    scale = max(target_width / source_width, target_height / source_height)
    resized_width = max(target_width, round(source_width * scale))
    resized_height = max(target_height, round(source_height * scale))
    resized = cv2.resize(
        rgb, (resized_width, resized_height), interpolation=cv2.INTER_AREA
    )
    left = (resized_width - target_width) // 2
    top = (resized_height - target_height) // 2
    return resized[top : top + target_height, left : left + target_width]


def decode_window(
    path: Path,
    *,
    clip_id: str,
    seed: int,
    num_frames: int,
    fps: float,
    height: int,
    width: int,
):
    import av
    import numpy as np
    import torch

    window_seconds = (num_frames - 1) / fps
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        duration = probe_duration_seconds(container, stream)
        if duration + 1e-3 < window_seconds:
            raise RuntimeError(
                f"{path.name} is too short: {duration:.3f}s < {window_seconds:.3f}s"
            )
        slack = max(0.0, duration - window_seconds - 1.0 / fps)
        start = stable_fraction(seed, clip_id) * slack
        target_times = [start + index / fps for index in range(num_frames)]

        # Long clips can be tens of minutes. Seek to a keyframe shortly before
        # the deterministic crop instead of decoding from the beginning.
        if start > 0 and stream.time_base is not None:
            seek_seconds = max(0.0, start - 2.0)
            seek_timestamp = int(seek_seconds / float(stream.time_base))
            container.seek(
                seek_timestamp, stream=stream, any_frame=False, backward=True
            )

        selected = []
        next_index = 0
        last_rgb = None
        for frame in container.decode(stream):
            if frame.time is None:
                continue
            if frame.time + 1e-6 < target_times[next_index]:
                continue
            rgb = frame.to_ndarray(format="rgb24")
            last_rgb = rgb
            while next_index < num_frames and frame.time + 1e-6 >= target_times[next_index]:
                selected.append(resize_center_crop(rgb, height, width))
                next_index += 1
            if next_index == num_frames:
                break
        if last_rgb is not None:
            while len(selected) < num_frames:
                selected.append(resize_center_crop(last_rgb, height, width))
        if len(selected) != num_frames:
            raise RuntimeError(f"Decoded only {len(selected)}/{num_frames} frames")

    frames = np.stack(selected, axis=0)
    pixel = torch.from_numpy(frames).permute(3, 0, 1, 2).float()
    return pixel.div_(127.5).sub_(1.0), start, duration


def iter_rank_rows(rows: list[dict[str, str]], rank: int, world_size: int) -> Iterable:
    for index, row in enumerate(rows):
        if index % world_size == rank:
            yield index, row


def atomic_torch_save(torch, payload: dict, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".part")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def validate_existing_shard(
    torch,
    destination: Path,
    *,
    row: dict[str, str],
    args: argparse.Namespace,
    expected_shape: tuple[int, int, int, int],
) -> None:
    try:
        item = torch.load(destination, map_location="cpu", weights_only=False)
        latent = item["latent"]
        preprocess = item["preprocess"]
    except Exception as exc:
        raise RuntimeError(
            f"Existing shard is unreadable: {destination}. Move it aside and rerun."
        ) from exc

    expected_metadata = {
        "clip_id": row["clip_id"],
        "prompt": row[args.caption_field].strip(),
        "source_video": Path(row["local_path"]).name,
    }
    actual_metadata = {key: item.get(key) for key in expected_metadata}
    if actual_metadata != expected_metadata:
        raise RuntimeError(
            f"Existing shard metadata mismatch: {destination}: "
            f"expected={expected_metadata}, actual={actual_metadata}"
        )
    expected_preprocess = {
        "seed": args.seed,
        "pixel_frames": args.num_frames,
        "fps": args.fps,
        "height": args.height,
        "width": args.width,
    }
    for key, expected in expected_preprocess.items():
        if preprocess.get(key) != expected:
            raise RuntimeError(
                f"Existing shard preprocess mismatch for {destination}: "
                f"{key}={preprocess.get(key)!r}, expected {expected!r}"
            )
    if not torch.is_tensor(latent) or tuple(latent.shape) != expected_shape:
        raise RuntimeError(
            f"Existing shard shape mismatch for {destination}: "
            f"{getattr(latent, 'shape', None)} != {expected_shape}"
        )
    if latent.dtype != torch.float16 or not bool(torch.isfinite(latent).all()):
        raise RuntimeError(
            f"Existing shard must be finite float16: {destination}, dtype={latent.dtype}"
        )


def validate_completed_file_set(output_dir: Path, plan: dict) -> None:
    expected = {sample["filename"] for sample in plan["samples"]}
    actual = {path.name for path in output_dir.glob("*.pt")}
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        raise RuntimeError(
            "Encoded shard set is incomplete or contaminated: "
            f"missing={missing[:8]} ({len(missing)} total), "
            f"unexpected={unexpected[:8]} ({len(unexpected)} total)"
        )


def main() -> int:
    args = parse_args()
    if args.num_frames < 1 or (args.num_frames - 1) % 4 != 0:
        raise ValueError(
            "--num-frames must have the form 4k+1 (for example 81, 249 or 321)"
        )
    if args.fps <= 0 or args.height <= 0 or args.width <= 0:
        raise ValueError("--fps, --height and --width must be positive")
    if args.limit < 0:
        raise ValueError("--limit must be non-negative")
    if args.height % 16 or args.width % 16:
        raise ValueError("--height and --width must be divisible by 16")

    rows = load_manifest(args.manifest.resolve(), args.split)
    if args.skip_too_short:
        window_seconds = (args.num_frames - 1) / args.fps
        kept_rows = []
        skipped = 0
        for row in rows:
            duration = row.get("duration", "").strip()
            if duration and float(duration) + 1e-3 < window_seconds:
                skipped += 1
                continue
            kept_rows.append(row)
        rows = kept_rows
        print(
            f"[duration filter] kept {len(rows)} rows and skipped {skipped} "
            f"shorter than {window_seconds:.3f}s")
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise RuntimeError("No manifest rows remain after split/duration filtering")
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    assigned = list(iter_rank_rows(rows, rank, world_size))
    print(
        f"[rank {rank}/{world_size}] {len(assigned)} of {len(rows)} clips; "
        f"{args.num_frames} frames at {args.fps:g}fps -> "
        f"{(args.num_frames - 1) // 4 + 1} Wan latents"
    )
    for index, row in assigned:
        resolve_video(args.videos_dir.resolve(), row)
        if not (row.get(args.caption_field) or "").strip():
            raise RuntimeError(f"Empty caption for {row['clip_id']}")
    if args.dry_run:
        print("[dry-run] Manifest, captions and assigned video paths are valid.")
        return 0

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Wan VAE encoding requires a CUDA GPU")
    torch.cuda.set_device(local_rank)
    distributed = world_size > 1
    if distributed:
        torch.distributed.init_process_group(backend="gloo", init_method="env://")

    plan = build_encode_plan(args, rows)
    if rank == 0:
        ensure_encode_plan(args.output_dir, plan)
    if distributed:
        torch.distributed.barrier()
    ensure_encode_plan(args.output_dir, plan)

    from utils.wan_wrapper import WanVAEWrapper

    device = torch.device("cuda", local_rank)
    vae = WanVAEWrapper().to(device)
    completed = 0
    expected_shape = tuple(plan["latent_shape"])
    for index, row in assigned:
        clip_id = row["clip_id"]
        destination = args.output_dir / output_name(index, clip_id)
        if destination.is_file() and destination.stat().st_size > 0:
            validate_existing_shard(
                torch,
                destination,
                row=row,
                args=args,
                expected_shape=expected_shape,
            )
            completed += 1
            continue
        video_path = resolve_video(args.videos_dir.resolve(), row)
        pixel, start, duration = decode_window(
            video_path,
            clip_id=clip_id,
            seed=args.seed,
            num_frames=args.num_frames,
            fps=args.fps,
            height=args.height,
            width=args.width,
        )
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            latent = vae.encode_to_latent(pixel.unsqueeze(0).to(device))[0]
        latent = latent.to(device="cpu", dtype=torch.float16).contiguous()
        if tuple(latent.shape) != expected_shape:
            raise RuntimeError(f"Unexpected latent shape for {clip_id}: {tuple(latent.shape)}")
        atomic_torch_save(
            torch,
            {
                "clip_id": clip_id,
                "prompt": row[args.caption_field].strip(),
                "latent": latent,
                "source_video": video_path.name,
                "preprocess": {
                    "seed": args.seed,
                    "start_seconds": start,
                    "source_duration_seconds": duration,
                    "pixel_frames": args.num_frames,
                    "fps": args.fps,
                    "height": args.height,
                    "width": args.width,
                },
            },
            destination,
        )
        completed += 1
        print(f"[rank {rank}] {completed}/{len(assigned)} {clip_id} -> {destination.name}")

    if distributed:
        torch.distributed.barrier()
    if rank == 0:
        validate_completed_file_set(args.output_dir, plan)
        atomic_json_write(
            args.output_dir / ENCODE_COMPLETE_NAME,
            {
                "schema_version": ENCODE_SCHEMA_VERSION,
                "complete": True,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "count": plan["count"],
                "latent_shape": plan["latent_shape"],
                "plan_sha256": plan["plan_sha256"],
                "plan_file": ENCODE_PLAN_NAME,
            },
        )
        print(
            f"[encode manifest] {plan['count']} verified shards; "
            f"plan_sha256={plan['plan_sha256']}"
        )
    if distributed:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    print(f"[rank {rank}] COMPLETE ({completed} shards)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted safely; rerun the same command to resume.", file=sys.stderr)
        raise SystemExit(130)
