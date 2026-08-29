#!/usr/bin/env python3
"""Range-download a tiny locked UltraVideo-Long subset for code debugging.

Unlike the production downloader, this script reads remote ZIP central
directories and fetches only selected members.  It is intended for a handful
of smoke-test clips, not for constructing the unbiased 5K training set.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path

from download_ultravideo_long_5k import (
    DEFAULT_FILES,
    api_tree,
    atomic_json,
    auth_headers,
    download_with_resume,
    load_locked_selection,
    load_rows,
    natural_key,
    resolve_url,
    selection_digest,
    write_manifests,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch a few locked UltraVideo-Long members via HTTP Range."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("dataset/UltraVideo-Long-debug")
    )
    parser.add_argument("--selection-ids", type=Path, required=True)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--resolution", choices=("960", "1920"), default="960")
    parser.add_argument("--min-duration", type=float, default=20.0)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=12)
    parser.add_argument("--initial-buffer-mb", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.count <= 0 or args.offset < 0:
        raise ValueError("--count must be positive and --offset non-negative")
    try:
        from remotezip import RemoteZip
    except ImportError as exc:
        raise RuntimeError("Install the pinned dependency: pip install remotezip==0.12.3") from exc

    root = args.output_dir.expanduser().resolve()
    metadata_dir = root / "metadata"
    manifest_dir = root / "manifests"
    videos_dir = root / "videos"
    state_dir = root / ".state"
    for directory in (metadata_dir, manifest_dir, videos_dir, state_dir):
        directory.mkdir(parents=True, exist_ok=True)

    top_tree = {entry["path"]: entry for entry in api_tree("", args.timeout)}
    for name in DEFAULT_FILES:
        entry = top_tree.get(name)
        download_with_resume(
            name,
            metadata_dir / name,
            expected_size=int(entry["size"]) if entry else None,
            retries=args.retries,
            timeout=args.timeout,
        )

    rows, original_fields = load_rows(metadata_dir / "long.csv")
    with args.selection_ids.expanduser().resolve().open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        locked_count = sum(1 for _ in csv.DictReader(handle))
    locked = load_locked_selection(
        rows,
        args.selection_ids.expanduser().resolve(),
        count=locked_count,
        min_duration=args.min_duration,
    )
    selected = locked[args.offset : args.offset + args.count]
    if len(selected) != args.count:
        raise RuntimeError("Requested debug slice extends beyond the locked manifest")
    # The debug pipeline consumes every selected item regardless of its formal
    # 5K train/validation assignment.
    for row in selected:
        row["source_split"] = row["split"]
        row["split"] = "train"
    write_manifests(selected, [*original_fields, "source_split"], manifest_dir)
    wanted = {row["clip_id"] for row in selected}
    digest = selection_digest(wanted)
    print(f"[debug selection] {len(wanted)} clips, sha256={digest}")

    archive_dir_name = f"clips_long_{args.resolution}"
    archives = [
        entry
        for entry in api_tree(archive_dir_name, args.timeout)
        if entry.get("type") == "file" and str(entry.get("path", "")).endswith(".zip")
    ]
    archives.sort(key=lambda entry: natural_key(str(entry["path"])))
    remaining = set(wanted)
    for archive_index, entry in enumerate(archives, start=1):
        if not remaining:
            break
        repo_path = str(entry["path"])
        print(f"[remote ZIP {archive_index}/{len(archives)}] index {repo_path}")
        try:
            with RemoteZip(
                resolve_url(repo_path),
                headers=auth_headers(),
                timeout=args.timeout,
                initial_buffer_size=args.initial_buffer_mb * 1024**2,
                support_suffix_range=True,
            ) as archive:
                matches = {
                    Path(info.filename).name: info
                    for info in archive.infolist()
                    if Path(info.filename).name in remaining and not info.is_dir()
                }
                for clip_id, info in matches.items():
                    destination = videos_dir / clip_id
                    if destination.is_file() and destination.stat().st_size == info.file_size:
                        remaining.remove(clip_id)
                        print(f"  [skip] {clip_id}")
                        continue
                    temporary = destination.with_suffix(destination.suffix + ".part")
                    temporary.unlink(missing_ok=True)
                    print(f"  [range extract] {clip_id} ({info.file_size / 1024**2:.1f} MiB)")
                    try:
                        with archive.open(info) as source, temporary.open("wb") as target:
                            shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
                            target.flush()
                            os.fsync(target.fileno())
                        if temporary.stat().st_size != info.file_size:
                            raise IOError(f"Size mismatch for {clip_id}")
                        os.replace(temporary, destination)
                        remaining.remove(clip_id)
                    except Exception:
                        temporary.unlink(missing_ok=True)
                        raise
        except Exception as exc:
            raise RuntimeError(
                f"Remote Range access failed for {repo_path}. Do not fall back to "
                "downloading the full archive during a smoke test."
            ) from exc

    if remaining:
        raise RuntimeError(f"Debug clips not found in official archives: {sorted(remaining)}")
    atomic_json(
        root / "COMPLETE.json",
        {
            "source": "APRIL-AIGC/UltraVideo-Long",
            "purpose": "range-downloaded code-debug subset",
            "count": len(wanted),
            "selection_sha256": digest,
            "videos": sorted(wanted),
        },
    )
    print(f"[COMPLETE] {len(wanted)} debug videos under {videos_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted safely; rerun the same command to resume.", file=sys.stderr)
        raise SystemExit(130)
