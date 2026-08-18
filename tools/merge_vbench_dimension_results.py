"""Merge independently persisted VBench dimension result files."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--expected-dimensions",
        default=(
            "subject_consistency,background_consistency,motion_smoothness,"
            "aesthetic_quality,imaging_quality,dynamic_degree"
        ),
        help="Comma-separated dimensions expected in the merged output.",
    )
    parser.add_argument("inputs", nargs="+")
    args = parser.parse_args()

    merged = {}
    for input_name in args.inputs:
        path = Path(input_name)
        data = json.loads(path.read_text(encoding="utf-8"))
        overlap = set(merged) & set(data)
        if overlap:
            raise ValueError(f"Duplicate dimensions {sorted(overlap)} in {path}")
        merged.update(data)

    expected = {
        name.strip()
        for name in args.expected_dimensions.split(",")
        if name.strip()
    }
    if set(merged) != expected:
        raise ValueError(
            f"Expected dimensions {sorted(expected)}, got {sorted(merged)}"
        )
    if "temporal_flickering" in merged:
        raise ValueError("temporal_flickering must not be included")

    Path(args.output).write_text(
        json.dumps(merged, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
