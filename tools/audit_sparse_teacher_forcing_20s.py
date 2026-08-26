#!/usr/bin/env python3
"""Audit the 20-second slow/fast sparse-target teacher-forcing contract."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from omegaconf import OmegaConf

from utils.teacher_forcing import (
    build_balanced_target_block_patterns,
    build_sparse_teacher_forcing_frame_mask,
    history_frame_indices,
)


EXPECTED_FINAL_HISTORY = (
    0, 1, 2,
    3, 9, 15, 21, 27, 33, 39, 45, 51, 57, 63, 69,
    75, 76, 77,
)


def audit_config(path: Path):
    config = OmegaConf.load(path)
    expected = {
        "teacher_forcing": True,
        "num_training_frames": 81,
        "num_frame_per_block": 3,
        "sparse_target_num_blocks": 12,
        "sparse_target_num_patterns": 9,
        "sparse_target_pattern_stride": 2,
        "model_kwargs.max_num_frames": 81,
        "model_kwargs.local_attn_size": 21,
        "model_kwargs.sink_size": 3,
        "model_kwargs.slow_context_size": 12,
        "model_kwargs.fast_context_size": 3,
        "model_kwargs.slow_context_stride_blocks": 2,
    }
    actual = {
        "teacher_forcing": bool(config.teacher_forcing),
        "num_training_frames": int(config.num_training_frames),
        "num_frame_per_block": int(config.num_frame_per_block),
        "sparse_target_num_blocks": int(config.sparse_target_num_blocks),
        "sparse_target_num_patterns": int(config.sparse_target_num_patterns),
        "sparse_target_pattern_stride": int(config.sparse_target_pattern_stride),
        "model_kwargs.max_num_frames": int(config.model_kwargs.max_num_frames),
        "model_kwargs.local_attn_size": int(config.model_kwargs.local_attn_size),
        "model_kwargs.sink_size": int(config.model_kwargs.sink_size),
        "model_kwargs.slow_context_size": int(config.model_kwargs.slow_context_size),
        "model_kwargs.fast_context_size": int(config.model_kwargs.fast_context_size),
        "model_kwargs.slow_context_stride_blocks": int(
            config.model_kwargs.slow_context_stride_blocks),
    }
    if actual != expected:
        raise AssertionError(f"config mismatch: expected={expected}, actual={actual}")
    return config


def audit_patterns_and_masks(config) -> None:
    patterns = build_balanced_target_block_patterns(
        num_frames=81,
        block_size=3,
        num_target_blocks=12,
        num_patterns=9,
        traversal_stride=2,
    )
    if len(patterns) != 9 or any(len(pattern) != 12 for pattern in patterns):
        raise AssertionError(f"unexpected target pattern bank: {patterns}")
    counts = Counter(block_start for pattern in patterns for block_start in pattern)
    expected_starts = set(range(0, 81, 3))
    if set(counts) != expected_starts or set(counts.values()) != {4}:
        raise AssertionError(f"target bank is not exactly balanced: {counts}")

    final_history = history_frame_indices(
        78,
        num_frames=81,
        block_size=3,
        sink_size=3,
        slow_size=12,
        fast_size=3,
        slow_stride_blocks=2,
    )
    if final_history != EXPECTED_FINAL_HISTORY:
        raise AssertionError(
            f"final history mismatch: {final_history} != {EXPECTED_FINAL_HISTORY}")

    for pattern in patterns:
        frame_mask = build_sparse_teacher_forcing_frame_mask(
            num_clean_frames=81,
            target_block_starts=pattern,
            block_size=3,
            sink_size=3,
            slow_size=12,
            fast_size=3,
            slow_stride_blocks=2,
        )
        if tuple(frame_mask.shape) != (117, 117):
            raise AssertionError(
                f"packed sequence must be clean81 + noisy36, got {frame_mask.shape}")
        if int(frame_mask.sum(dim=1).max()) > 21:
            raise AssertionError("a query exceeds the strict 21-frame visible budget")

        for packed_index, block_start in enumerate(pattern):
            query_index = 81 + packed_index * 3
            expected_history = set(history_frame_indices(
                block_start,
                num_frames=81,
                block_size=3,
                sink_size=3,
                slow_size=12,
                fast_size=3,
                slow_stride_blocks=2,
            ))
            own_noisy = set(range(query_index, query_index + 3))
            actual = set(frame_mask[query_index].nonzero().flatten().tolist())
            if actual != expected_history | own_noisy:
                raise AssertionError(
                    f"target block {block_start} mask mismatch: actual={actual}")
            if any(frame_mask[query_index, block_start:block_start + 3].tolist()):
                raise AssertionError(
                    f"target block {block_start} leaked its clean current chunk")

    full_target_frames = 81 + 81
    sparse_target_frames = 81 + 12 * 3
    reduction = 1.0 - sparse_target_frames / full_target_frames
    print(
        "SPARSE_TEACHER_FORCING_20S_AUDIT_OK:"
        f"patterns=9:targets_per_step=12/27:packed_frames={sparse_target_frames}:"
        f"sequence_frame_reduction={reduction:.3%}"
    )


def audit_inference_cache_policy() -> None:
    original_current_device = None
    if not torch.cuda.is_available():
        # Wan's T5 module evaluates torch.cuda.current_device() in a constructor
        # default at import time even though this policy audit allocates CPU-only
        # tensors. Supply a harmless index while an AutoDL instance is parked.
        original_current_device = torch.cuda.current_device
        torch.cuda.current_device = lambda: 0
    try:
        from wan.modules.causal_model import CausalWanSelfAttention
    finally:
        if original_current_device is not None:
            torch.cuda.current_device = original_current_device

    attention = CausalWanSelfAttention(
        dim=8,
        num_heads=1,
        local_attn_size=21,
        sink_size=3,
        slow_context_size=12,
        fast_context_size=3,
        slow_context_stride_blocks=2,
    )
    cache = {
        "k": torch.zeros(1, 21, 1, 1),
        "v": torch.zeros(1, 21, 1, 1),
        "global_end_index": torch.zeros(1, dtype=torch.long),
        "local_end_index": torch.zeros(1, dtype=torch.long),
    }
    _, current_size = attention._ensure_slowfast_cache_metadata(cache, 1)
    scratch_start = 18
    for block_start in range(0, 78, 3):
        current_frames = torch.arange(block_start, block_start + 3)
        cache["source_frames"][scratch_start:21] = current_frames
        cache["k"][:, scratch_start:21, 0, 0] = current_frames
        cache["v"][:, scratch_start:21, 0, 0] = current_frames
        attention._commit_slowfast_cache(
            cache,
            frame_seqlen=1,
            current_size=current_size,
            current_start_frame=block_start,
            num_current_frames=3,
        )
        next_start = block_start + 3
        actual = tuple(
            int(value) for value in cache["source_frames"][:18].tolist()
            if value >= 0)
        expected = history_frame_indices(
            next_start,
            num_frames=81,
            block_size=3,
            sink_size=3,
            slow_size=12,
            fast_size=3,
            slow_stride_blocks=2,
        )
        if actual != expected:
            raise AssertionError(
                f"inference cache mismatch before block {next_start}: "
                f"actual={actual}, expected={expected}")
    print("SLOWFAST_CACHE_POLICY_AUDIT_OK:preview_scratch_then_clean_commit")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = audit_config(args.config)
    audit_patterns_and_masks(config)
    audit_inference_cache_policy()


if __name__ == "__main__":
    main()
