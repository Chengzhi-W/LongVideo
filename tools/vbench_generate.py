"""Distributed VBench sampling for Causal Forcing checkpoints.

The manifest keeps the short, canonical VBench prompt for the filename and the
Wan-expanded prompt for conditioning.  Tasks are sharded exactly across ranks
without DistributedSampler padding, and every checkpoint sees identical noise.
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from einops import rearrange
from omegaconf import OmegaConf
from torchvision.io import write_video
from tqdm import tqdm

from demo_utils.memory import DynamicSwapInstaller, get_cuda_free_memory_gb, gpu
from pipeline import CausalInferencePipeline
from utils.misc import set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_folder", required=True)
    parser.add_argument("--num_output_frames", type=int, default=63)
    parser.add_argument("--samples_per_prompt", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument(
        "--seed_mode",
        choices=["per_task", "prompt_batch_stream"],
        default="per_task",
        help=(
            "per_task reproduces the original helper's independently reset seeds; "
            "prompt_batch_stream seeds once, then draws all samples for each prompt "
            "as one tensor, matching the VBench sampling pseudocode."
        ),
    )
    parser.add_argument(
        "--rank_seed_offset",
        action="store_true",
        help="Use base_seed + rank for prompt_batch_stream instead of the same seed on every rank.",
    )
    parser.add_argument("--suite", default="", help="Only sample one manifest suite.")
    parser.add_argument("--limit_prompts", type=int, default=0)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--limit_tasks", type=int, default=0)
    return parser.parse_args()


def distributed_context():
    if "LOCAL_RANK" not in os.environ:
        return 0, 1, torch.device("cuda")
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, dist.get_world_size(), torch.device(f"cuda:{local_rank}")


def load_generator(pipeline, checkpoint_path, use_ema):
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = state["generator_ema" if use_ema else "generator"]
    fixed = {}
    for key, value in state_dict.items():
        if key.startswith("model._fsdp_wrapped_module."):
            key = key.replace("model._fsdp_wrapped_module.", "model.", 1)
        fixed[key] = value
    incompatible = pipeline.generator.load_state_dict(fixed, strict=False)
    if incompatible.missing_keys:
        raise RuntimeError(f"Missing generator keys: {incompatible.missing_keys[:20]}")
    if incompatible.unexpected_keys:
        print(f"Ignoring {len(incompatible.unexpected_keys)} unexpected checkpoint keys")


def main():
    args = parse_args()
    rank, world_size, device = distributed_context()
    config = OmegaConf.merge(
        OmegaConf.load("configs/default_config.yaml"),
        OmegaConf.load(args.config_path),
    )
    if not hasattr(config, "denoising_step_list"):
        raise ValueError("VBench sampler expects a few-step causal checkpoint")

    pipeline = CausalInferencePipeline(config, device=device)
    load_generator(pipeline, args.checkpoint_path, args.use_ema)
    pipeline = pipeline.to(dtype=torch.bfloat16)
    low_memory = get_cuda_free_memory_gb(gpu) < 40
    if low_memory:
        DynamicSwapInstaller.install_model(pipeline.text_encoder, device=gpu)
    else:
        pipeline.text_encoder.to(device=device)
    pipeline.generator.to(device=device)
    pipeline.vae.to(device=device)
    pipeline.eval()
    torch.set_grad_enabled(False)

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if args.suite:
        manifest = [item for item in manifest if item.get("suite") == args.suite]
    if args.limit_prompts > 0:
        manifest = manifest[: args.limit_prompts]

    output_dir = Path(args.output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.seed_mode == "prompt_batch_stream":
        # VBench's reference sampling pseudocode seeds once, then samples five
        # videos per prompt.  Draw the complete five-sample tensor before
        # splitting it into inference micro-batches so batch_size=4 does not
        # change the sampled noise.  We intentionally draw noise even for an
        # already completed prompt, which makes resumed runs preserve the RNG
        # position of an uninterrupted run.
        stream_seed = args.base_seed + (rank if args.rank_seed_offset else 0)
        if dist.is_initialized():
            local_seed = torch.tensor([stream_seed], device=device, dtype=torch.int64)
            gathered_seeds = [torch.zeros_like(local_seed) for _ in range(world_size)]
            dist.all_gather(gathered_seeds, local_seed)
            rank_seeds = [int(seed.item()) for seed in gathered_seeds]
        else:
            rank_seeds = [stream_seed]
        if rank == 0:
            print(f"SEED_AUDIT:rank_seeds={rank_seeds}", flush=True)
        if not args.rank_seed_offset and any(seed != args.base_seed for seed in rank_seeds):
            raise RuntimeError(
                f"Strict base-seed mode expected every rank to use {args.base_seed}, "
                f"but observed {rank_seeds}"
            )
        set_seed(stream_seed)
        rank_prompts = list(enumerate(manifest))[rank::world_size]
        iterator = tqdm(rank_prompts, disable=rank != 0, desc=f"rank{rank}")
        for _, item in iterator:
            output_paths = [
                output_dir / f"{item['prompt']}-{sample_index}.mp4"
                for sample_index in range(args.samples_per_prompt)
            ]
            all_noise = torch.randn(
                [args.samples_per_prompt, args.num_output_frames, 16, 60, 104],
                device=device,
                dtype=torch.bfloat16,
            )
            missing = [
                sample_index
                for sample_index, output_path in enumerate(output_paths)
                if not output_path.exists() or output_path.stat().st_size == 0
            ]
            for start in range(0, len(missing), args.batch_size):
                sample_indices = missing[start : start + args.batch_size]
                real_batch_size = len(sample_indices)
                padded_indices = list(sample_indices)
                if padded_indices and real_batch_size < args.batch_size:
                    padded_indices.extend(
                        [padded_indices[-1]] * (args.batch_size - real_batch_size)
                    )
                if not padded_indices:
                    continue
                conditioning_prompt = item.get("extended_prompt") or item["prompt"]
                video = pipeline.inference(
                    noise=all_noise[padded_indices],
                    text_prompts=[conditioning_prompt] * len(padded_indices),
                    return_latents=False,
                )
                frames = rearrange(video, "b t c h w -> b t h w c").cpu()
                for frame_batch, sample_index in zip(
                    frames[:real_batch_size], sample_indices
                ):
                    write_video(
                        str(output_paths[sample_index]), 255.0 * frame_batch, fps=16
                    )
                pipeline.vae.model.clear_cache()

        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        return

    tasks = []
    for prompt_index, item in enumerate(manifest):
        for sample_index in range(args.samples_per_prompt):
            tasks.append((prompt_index, sample_index, item))
    if args.limit_tasks > 0:
        tasks = tasks[: args.limit_tasks]

    rank_tasks = tasks[rank::world_size]
    iterator = tqdm(rank_tasks, disable=rank != 0, desc=f"rank{rank}")

    pending = []
    for task in rank_tasks:
        prompt_index, sample_index, item = task
        output_path = output_dir / f"{item['prompt']}-{sample_index}.mp4"
        if not output_path.exists() or output_path.stat().st_size == 0:
            pending.append(task)

    batches = [pending[i:i + args.batch_size] for i in range(0, len(pending), args.batch_size)]
    iterator = tqdm(batches, disable=rank != 0, desc=f"rank{rank}")
    for batch in iterator:
        # The causal attention cache is allocated for the configured batch size.
        # Pad only the final short batch, then discard the padded outputs below.
        real_batch_size = len(batch)
        padded_batch = list(batch)
        if padded_batch and real_batch_size < args.batch_size:
            padded_batch.extend([padded_batch[-1]] * (args.batch_size - real_batch_size))
        conditioning_prompts = []
        output_paths = []
        noises = []
        for prompt_index, sample_index, item in padded_batch:
            conditioning_prompts.append(item.get("extended_prompt") or item["prompt"])
            output_paths.append(output_dir / f"{item['prompt']}-{sample_index}.mp4")
            task_seed = args.base_seed + prompt_index * args.samples_per_prompt + sample_index
            set_seed(task_seed)
            noises.append(torch.randn(
                [1, args.num_output_frames, 16, 60, 104],
                device=device,
                dtype=torch.bfloat16,
            ))
        noise = torch.cat(noises, dim=0)
        video = pipeline.inference(
            noise=noise,
            text_prompts=conditioning_prompts,
            return_latents=False,
        )
        frames = rearrange(video, "b t c h w -> b t h w c").cpu()
        for frame_batch, output_path in zip(frames[:real_batch_size], output_paths[:real_batch_size]):
            write_video(str(output_path), 255.0 * frame_batch, fps=16)
        pipeline.vae.model.clear_cache()

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
