#!/usr/bin/env python3
"""Verify that a fine-tuned AR-teacher checkpoint is DMD-loader compatible."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def _extract_state(path: Path) -> dict[str, torch.Tensor]:
    value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise TypeError(f"checkpoint is not a dictionary: {path}")
    for key in ("generator", "generator_ema", "model"):
        if key in value:
            value = value[key]
            break
    if not isinstance(value, dict):
        raise TypeError(f"model state is not a dictionary: {path}")
    return {
        key.replace("model._fsdp_wrapped_module.", "model.", 1): tensor
        for key, tensor in value.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--tuned", type=Path, required=True)
    args = parser.parse_args()

    base = _extract_state(args.base)
    tuned = _extract_state(args.tuned)
    if base.keys() != tuned.keys():
        missing = sorted(base.keys() - tuned.keys())
        unexpected = sorted(tuned.keys() - base.keys())
        raise AssertionError(
            f"state-key mismatch: missing={missing[:10]}, unexpected={unexpected[:10]}")

    changed = 0
    for key, tuned_tensor in tuned.items():
        if not torch.isfinite(tuned_tensor).all():
            raise AssertionError(f"non-finite values in tuned tensor: {key}")
        base_tensor = base[key]
        if base_tensor.shape != tuned_tensor.shape:
            raise AssertionError(
                f"shape mismatch for {key}: {base_tensor.shape} != {tuned_tensor.shape}")
        changed += int(not torch.equal(base_tensor, tuned_tensor))

    if changed == 0:
        raise AssertionError("no model tensor changed during the smoke fine-tune")
    print(
        "AR_TEACHER_CHECKPOINT_AUDIT_OK:"
        f"keys={len(tuned)}:changed={changed}:all_finite=true")


if __name__ == "__main__":
    main()
