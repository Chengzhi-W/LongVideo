"""Build a manifest from aligned original and Wan seed42 VBench prompts."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vbench_root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--suites",
        nargs="+",
        default=["subject_consistency", "scene", "overall_consistency"],
    )
    args = parser.parse_args()

    root = Path(args.vbench_root) / "prompts"
    originals = (root / "all_dimension.txt").read_text(encoding="utf-8").splitlines()
    expanded = (
        root
        / "augmented_prompts/Wan2.1-T2V-1.3B/all_dimension_aug_wanx_seed42.txt"
    ).read_text(encoding="utf-8").splitlines()
    if len(originals) != len(expanded):
        raise ValueError("Original and Wan-expanded VBench prompt files are misaligned")
    expanded_by_prompt = dict(zip(originals, expanded))

    manifest = []
    seen = set()
    suites = args.suites
    for suite in suites:
        prompts = (root / "prompts_per_dimension" / f"{suite}.txt").read_text(
            encoding="utf-8"
        ).splitlines()
        for prompt in prompts:
            if prompt in seen:
                continue
            seen.add(prompt)
            manifest.append(
                {
                    "prompt": prompt,
                    "extended_prompt": expanded_by_prompt[prompt],
                    "suite": suite,
                }
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    counts = {suite: sum(item["suite"] == suite for item in manifest) for suite in suites}
    print(f"Wrote {len(manifest)} unique prompts to {output}: {counts}")


if __name__ == "__main__":
    main()
