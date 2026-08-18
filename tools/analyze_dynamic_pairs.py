#!/usr/bin/env python3
"""Compare official Dynamic Degree margins and global-flow proxies for video pairs."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from easydict import EasyDict as edict

from vbench.dynamic_degree import DynamicDegree
from vbench.third_party.RAFT.core.utils_core.utils import InputPadder


def analyze(dynamic, video_path: str):
    frames = dynamic.get_frames(video_path)
    dynamic.set_params(frames[0], len(frames))
    top5_scores = []
    median_scores = []
    with torch.no_grad():
        for image1, image2 in zip(frames[:-1], frames[1:]):
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)
            _, flow = dynamic.model(image1, image2, iters=20, test_mode=True)
            flow = flow[0].permute(1, 2, 0).float().cpu().numpy()
            rad = np.sqrt(np.square(flow[..., 0]) + np.square(flow[..., 1]))
            cut = int(rad.size * 0.05)
            top5_scores.append(float(np.mean(np.sort(rad.reshape(-1))[-cut:])))
            median_scores.append(float(np.median(rad)))

    top5 = np.asarray(top5_scores)
    median = np.asarray(median_scores)
    threshold = float(dynamic.params["thres"])
    required = int(dynamic.params["count_num"])
    above = int(np.sum(top5 > threshold))
    return {
        "video": str(video_path),
        "sampled_frames": len(frames),
        "flow_pairs": len(top5),
        "threshold": threshold,
        "required_pairs": required,
        "pairs_above_threshold": above,
        "official_dynamic": above >= required,
        "top5_mean": float(top5.mean()),
        "top5_median": float(np.median(top5)),
        "top5_p90": float(np.percentile(top5, 90)),
        "global_median_mean": float(median.mean()),
        "global_median_p90": float(np.percentile(median, 90)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("videos", nargs="+")
    parser.add_argument(
        "--model",
        default="/root/.cache/vbench/raft_model/models/raft-things.pth",
    )
    args = parser.parse_args()

    cfg = edict(
        {
            "model": args.model,
            "small": False,
            "mixed_precision": False,
            "alternate_corr": False,
        }
    )
    dynamic = DynamicDegree(cfg, torch.device("cuda:0"))
    results = [analyze(dynamic, path) for path in args.videos]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
