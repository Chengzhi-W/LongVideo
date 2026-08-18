"""Calculate the six-dimension VBench Quality Score with TF omitted.

Scores are first normalized with the official VBench bounds.  The remaining
weights are renormalized to sum to one; dynamic degree retains weight 0.5.
"""

import argparse
import json
from pathlib import Path


BOUNDS = {
    "subject_consistency": (0.1462, 1.0),
    "background_consistency": (0.2615, 1.0),
    "motion_smoothness": (0.7060, 0.9975),
    "aesthetic_quality": (0.0, 1.0),
    "imaging_quality": (0.0, 1.0),
    "dynamic_degree": (0.0, 1.0),
}
WEIGHTS = {key: (0.5 if key == "dynamic_degree" else 1.0) for key in BOUNDS}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", help="VBench per-dimension JSON files")
    args = parser.parse_args()
    raw = {}
    for result_path in args.results:
        data = json.loads(Path(result_path).read_text(encoding="utf-8"))
        for key, value in data.items():
            if key in BOUNDS:
                raw[key] = float(value[0] if isinstance(value, list) else value)
    missing = sorted(set(BOUNDS) - set(raw))
    if missing:
        raise ValueError(f"Missing dimensions: {missing}")

    normalized = {
        key: (raw[key] - BOUNDS[key][0]) / (BOUNDS[key][1] - BOUNDS[key][0])
        for key in BOUNDS
    }
    denominator = sum(WEIGHTS.values())
    quality = sum(normalized[key] * WEIGHTS[key] for key in BOUNDS) / denominator
    print(json.dumps({"raw": raw, "normalized": normalized,
                      "quality_score_no_temporal_flickering": quality}, indent=2))


if __name__ == "__main__":
    main()
