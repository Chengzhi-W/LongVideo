#!/usr/bin/env python3
"""Audit the long-video AR teacher's sink+recent attention contract."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf

from wan.modules.causal_model import CausalWanModel


def _allowed(mask_mod, query: int, key: int) -> bool:
    index = lambda value: torch.tensor(value, dtype=torch.int64)
    return bool(mask_mod(index(0), index(0), index(query), index(key)).item())


def _allowed_keys(mask_mod, query: int, length: int) -> set[int]:
    return {key for key in range(length) if _allowed(mask_mod, query, key)}


def audit_mask() -> None:
    # One token per latent frame makes the exact frame-level topology observable.
    frames = 63
    clean_end = frames
    total = frames * 2
    mask = CausalWanModel._prepare_teacher_forcing_mask(
        "cpu",
        num_frames=frames,
        frame_seqlen=1,
        num_frame_per_block=3,
        local_attn_size=21,
        sink_size=3,
    )
    mask_mod = mask.mask_mod

    cases = {
        # First target has no clean history and can only self-attend inside its
        # three-frame noisy block.
        clean_end: set(range(clean_end, clean_end + 3)),
        # At an intermediate target: sink 0..2 + recent 15..29 + current 30..32.
        clean_end + 30: set(range(0, 3)) | set(range(15, 30)) |
        set(range(clean_end + 30, clean_end + 33)),
        # Final target: sink 0..2 + recent 45..59 + current 60..62.
        clean_end + 60: set(range(0, 3)) | set(range(45, 60)) |
        set(range(clean_end + 60, clean_end + 63)),
        # Committing the final clean block uses sink 0..2 + recent/current 45..62.
        60: set(range(0, 3)) | set(range(45, 63)),
    }
    for query, expected in cases.items():
        actual = _allowed_keys(mask_mod, query, total)
        if actual != expected:
            raise AssertionError(
                f"mask mismatch for query={query}: "
                f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}")


def audit_config(path: Path) -> None:
    config = OmegaConf.load(path)
    expected = {
        "teacher_forcing": True,
        "num_training_frames": 63,
        "num_frame_per_block": 3,
        "model_kwargs.max_num_frames": 63,
        "model_kwargs.local_attn_size": 21,
        "model_kwargs.sink_size": 3,
    }
    actual = {
        "teacher_forcing": bool(config.teacher_forcing),
        "num_training_frames": int(config.num_training_frames),
        "num_frame_per_block": int(config.num_frame_per_block),
        "model_kwargs.max_num_frames": int(config.model_kwargs.max_num_frames),
        "model_kwargs.local_attn_size": int(config.model_kwargs.local_attn_size),
        "model_kwargs.sink_size": int(config.model_kwargs.sink_size),
    }
    if actual != expected:
        raise AssertionError(f"config mismatch: expected={expected}, actual={actual}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    audit_config(args.config)
    audit_mask()
    print("SINK_RECENT_TEACHER_FORCING_AUDIT_OK")


if __name__ == "__main__":
    main()
