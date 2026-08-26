"""Utilities for sparse-target autoregressive teacher forcing.

The production 20-second layout keeps a strict 21-frame visible budget:

    sink3 + slow12 + fast3 + current3

The current block is packed separately from the clean history.  This module is
kept free of model code so the sampling schedule and attention semantics can be
audited without loading Wan checkpoints.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch


def _validate_layout(
    *,
    num_frames: int,
    block_size: int,
    sink_size: int,
    slow_size: int,
    fast_size: int,
    slow_stride_blocks: int,
) -> None:
    values = {
        "num_frames": num_frames,
        "block_size": block_size,
        "slow_stride_blocks": slow_stride_blocks,
    }
    for name, value in values.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    for name, value in {
        "sink_size": sink_size,
        "slow_size": slow_size,
        "fast_size": fast_size,
    }.items():
        if value < 0:
            raise ValueError(f"{name} must be non-negative, got {value}")
    if num_frames % block_size:
        raise ValueError(
            f"num_frames={num_frames} must be divisible by block_size={block_size}")
    if fast_size % block_size:
        raise ValueError(
            f"fast_size={fast_size} must be divisible by block_size={block_size}")


def history_frame_indices(
    block_start: int,
    *,
    num_frames: int,
    block_size: int,
    sink_size: int,
    slow_size: int,
    fast_size: int,
    slow_stride_blocks: int,
) -> tuple[int, ...]:
    """Return clean history visible while predicting ``block_start``.

    Slow anchors begin immediately after the sink and retain one latent frame
    every ``slow_stride_blocks`` chunks.  Frames still covered by the fast
    recent window are excluded from slow memory, so no frame consumes two slots.
    """
    _validate_layout(
        num_frames=num_frames,
        block_size=block_size,
        sink_size=sink_size,
        slow_size=slow_size,
        fast_size=fast_size,
        slow_stride_blocks=slow_stride_blocks,
    )
    if block_start < 0 or block_start >= num_frames:
        raise ValueError(
            f"block_start must be in [0, {num_frames}), got {block_start}")
    if block_start % block_size:
        raise ValueError(
            f"block_start={block_start} must align to block_size={block_size}")

    sink_end = min(sink_size, block_start)
    sink = list(range(sink_end))

    fast_start = max(sink_size, block_start - fast_size)
    fast = list(range(fast_start, block_start))

    stride_frames = slow_stride_blocks * block_size
    slow_candidates = list(range(sink_size, fast_start, stride_frames))
    slow = slow_candidates[-slow_size:] if slow_size else []

    history = tuple(sink + slow + fast)
    if len(history) != len(set(history)):
        raise RuntimeError(f"history layout contains duplicate frames: {history}")
    return history


def build_balanced_target_block_patterns(
    *,
    num_frames: int,
    block_size: int,
    num_target_blocks: int,
    num_patterns: int,
    traversal_stride: int = 2,
) -> tuple[tuple[int, ...], ...]:
    """Build a small balanced bank of target-block start indices.

    Each pattern selects complete chunks.  The modular traversal spreads nearby
    selections over the 20-second timeline, while cycling the start offset makes
    every one of the 27 chunks appear equally often in the default 9x12 bank.
    Keeping a small bank also bounds the number of FlexAttention masks compiled
    by each rank.
    """
    if num_frames <= 0 or block_size <= 0:
        raise ValueError("num_frames and block_size must be positive")
    if num_frames % block_size:
        raise ValueError(
            f"num_frames={num_frames} must be divisible by block_size={block_size}")
    num_blocks = num_frames // block_size
    if not 0 < num_target_blocks <= num_blocks:
        raise ValueError(
            f"num_target_blocks must be in [1, {num_blocks}], got {num_target_blocks}")
    if num_patterns <= 0:
        raise ValueError(f"num_patterns must be positive, got {num_patterns}")
    if math.gcd(traversal_stride, num_blocks) != 1:
        raise ValueError(
            f"traversal_stride={traversal_stride} must be coprime with "
            f"num_blocks={num_blocks}")
    if (num_patterns * num_target_blocks) % num_blocks:
        raise ValueError(
            "num_patterns * num_target_blocks must be divisible by num_blocks "
            "to give every chunk equal supervision")

    traversal = tuple((index * traversal_stride) % num_blocks
                      for index in range(num_blocks))
    patterns = []
    for pattern_index in range(num_patterns):
        offset = (pattern_index * num_target_blocks) % num_blocks
        block_ids = [
            traversal[(offset + target_index) % num_blocks]
            for target_index in range(num_target_blocks)
        ]
        patterns.append(tuple(sorted(block_id * block_size for block_id in block_ids)))

    appearances = [0] * num_blocks
    for pattern in patterns:
        if len(pattern) != len(set(pattern)):
            raise RuntimeError(f"target pattern contains duplicates: {pattern}")
        for block_start in pattern:
            appearances[block_start // block_size] += 1
    if len(set(appearances)) != 1:
        raise RuntimeError(f"target pattern bank is not balanced: {appearances}")
    return tuple(patterns)


def expand_target_blocks(
    target_block_starts: Iterable[int],
    *,
    num_frames: int,
    block_size: int,
) -> tuple[int, ...]:
    """Expand aligned target starts into sorted latent-frame indices."""
    starts = tuple(sorted(int(value) for value in target_block_starts))
    if len(starts) != len(set(starts)):
        raise ValueError(f"target block starts must be unique, got {starts}")
    frames = []
    for start in starts:
        if start < 0 or start + block_size > num_frames or start % block_size:
            raise ValueError(
                f"invalid target block [{start}, {start + block_size}) for "
                f"num_frames={num_frames}, block_size={block_size}")
        frames.extend(range(start, start + block_size))
    return tuple(frames)


def build_sparse_teacher_forcing_frame_mask(
    *,
    num_clean_frames: int,
    target_block_starts: Sequence[int],
    block_size: int,
    sink_size: int,
    slow_size: int,
    fast_size: int,
    slow_stride_blocks: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Return the frame-level mask for ``[clean][packed noisy targets]``.

    Clean blocks are computed with the same sparse history used at inference.
    A packed noisy block sees that clean history plus its own noisy chunk, never
    the clean copy of the current chunk.
    """
    _validate_layout(
        num_frames=num_clean_frames,
        block_size=block_size,
        sink_size=sink_size,
        slow_size=slow_size,
        fast_size=fast_size,
        slow_stride_blocks=slow_stride_blocks,
    )
    starts = tuple(sorted(int(value) for value in target_block_starts))
    target_frames = expand_target_blocks(
        starts, num_frames=num_clean_frames, block_size=block_size)
    num_target_frames = len(target_frames)
    total_frames = num_clean_frames + num_target_frames
    allowed = torch.zeros(
        (total_frames, total_frames), dtype=torch.bool, device=device)

    for block_start in range(0, num_clean_frames, block_size):
        block_end = block_start + block_size
        history = history_frame_indices(
            block_start,
            num_frames=num_clean_frames,
            block_size=block_size,
            sink_size=sink_size,
            slow_size=slow_size,
            fast_size=fast_size,
            slow_stride_blocks=slow_stride_blocks,
        )
        visible = history + tuple(range(block_start, block_end))
        allowed[block_start:block_end, list(visible)] = True

    noisy_offset = num_clean_frames
    for packed_block_index, block_start in enumerate(starts):
        query_start = noisy_offset + packed_block_index * block_size
        query_end = query_start + block_size
        history = history_frame_indices(
            block_start,
            num_frames=num_clean_frames,
            block_size=block_size,
            sink_size=sink_size,
            slow_size=slow_size,
            fast_size=fast_size,
            slow_stride_blocks=slow_stride_blocks,
        )
        allowed[query_start:query_end, list(history)] = True
        allowed[query_start:query_end, query_start:query_end] = True

    return allowed
