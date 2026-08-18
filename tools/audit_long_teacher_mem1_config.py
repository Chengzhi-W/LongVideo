#!/usr/bin/env python3
"""Reject accidental gates or Long-vs-Local settings in the main DMD config."""

from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    expected = {
        "generator_ckpt": "checkpoints/chunkwise/causal_ode.pt",
        "real_name": "Wan2.1-T2V-1.3B",
        "use_memory_dmd": True,
        "ar_teacher_name": "Wan2.1-T2V-1.3B",
        "ar_teacher_ckpt": (
            "checkpoints/chunkwise/ar_diffusion_long_sink_recent.pt"),
        "memory_control_type": "empty",
        "memory_history_layout": "rolling_sink_recent",
        "memory_sink_frames": 3,
        "memory_dmd_scale": 1.0,
        "memory_joint_normalization": False,
        "ar_teacher_kwargs.local_attn_size": 21,
        "ar_teacher_kwargs.sink_size": 3,
        "ar_teacher_kwargs.max_num_frames": 63,
        "ar_teacher_kwargs.frame_seq_length": 1560,
    }
    actual = {
        "generator_ckpt": str(config.generator_ckpt),
        "real_name": str(config.real_name),
        "use_memory_dmd": bool(config.use_memory_dmd),
        "ar_teacher_name": str(config.ar_teacher_name),
        "ar_teacher_ckpt": str(config.ar_teacher_ckpt),
        "memory_control_type": str(config.memory_control_type),
        "memory_history_layout": str(config.memory_history_layout),
        "memory_sink_frames": int(config.memory_sink_frames),
        "memory_dmd_scale": float(config.memory_dmd_scale),
        "memory_joint_normalization": bool(config.memory_joint_normalization),
        "ar_teacher_kwargs.local_attn_size": int(
            config.ar_teacher_kwargs.local_attn_size),
        "ar_teacher_kwargs.sink_size": int(config.ar_teacher_kwargs.sink_size),
        "ar_teacher_kwargs.max_num_frames": int(
            config.ar_teacher_kwargs.max_num_frames),
        "ar_teacher_kwargs.frame_seq_length": int(
            config.ar_teacher_kwargs.frame_seq_length),
    }
    if actual != expected:
        raise AssertionError(f"config mismatch: expected={expected}, actual={actual}")
    if "memory_max_timestep" in config:
        raise AssertionError("main experiment must not contain memory_max_timestep")
    if "memory_full_frames" in config or "memory_local_frames" in config:
        raise AssertionError("main experiment must not enable Long-vs-Local")
    print("LONG_TEACHER_MEM1_UNGATED_AUDIT_OK")


if __name__ == "__main__":
    main()
