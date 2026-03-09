#!/usr/bin/env python3
# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""
Evaluate a fine-tuned LingBot-VA checkpoint on LIBERO benchmark suites.

This script:
  1. Loads the fine-tuned model into a VA_Server instance.
  2. Connects to the LIBERO simulation environment.
  3. Runs closed-loop rollouts on all tasks in the specified suites.
  4. Reports per-task and per-suite success rates.

Requirements:
  - LIBERO package:  pip install libero
  - RoboSuite:       pip install robosuite
  - LIBERO benchmark data downloaded

Usage:
    python evaluation/libero/eval_libero.py \
        --config-name libero_a2a \
        --checkpoint checkpoints/libero-a2a-ft/step_004000 \
        --suites libero_spatial libero_object libero_goal libero_long \
        --n-eval 20 \
        --save-dir results/libero_eval
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "wan_va"))

from wan_va.configs import VA_CONFIGS
from wan_va.distributed.util import init_distributed
from wan_va.utils import init_logger, logger


def load_model_from_checkpoint(config, checkpoint_dir: str, device: torch.device):
    """Load VA_Server and override with fine-tuned weights."""
    from wan_va.wan_va_server import VA_Server

    server = VA_Server(config)

    ckpt_dir = Path(checkpoint_dir)
    if (ckpt_dir / "transformer.pt").exists():
        state = torch.load(ckpt_dir / "transformer.pt", map_location="cpu", weights_only=True)
        server.transformer.load_state_dict(state, strict=False)
        logger.info(f"Loaded transformer from {ckpt_dir / 'transformer.pt'}")

    if server.action_encoder is not None and (ckpt_dir / "action_encoder.pt").exists():
        server.action_encoder.load_state_dict(
            torch.load(ckpt_dir / "action_encoder.pt", map_location="cpu", weights_only=True)
        )
        logger.info("Loaded action_encoder")
    if server.action_decoder is not None and (ckpt_dir / "action_decoder.pt").exists():
        server.action_decoder.load_state_dict(
            torch.load(ckpt_dir / "action_decoder.pt", map_location="cpu", weights_only=True)
        )
        logger.info("Loaded action_decoder")

    return server


def get_libero_tasks(suite_name: str):
    """Return list of (task_name, task_description) for a LIBERO suite."""
    try:
        from libero.libero import benchmark
        bm = benchmark.get_benchmark(suite_name)()
        tasks = []
        for i in range(bm.n_tasks):
            task = bm.get_task(i)
            tasks.append((task.name, task.language))
        return tasks, bm
    except ImportError:
        logger.error("LIBERO package not installed. Install with: pip install libero")
        raise


def create_env(bm, task_idx: int, camera_height: int = 256, camera_width: int = 256):
    """Create a LIBERO environment for a specific task."""
    from libero.libero.envs import OffScreenRenderEnv
    task = bm.get_task(task_idx)
    env_args = {
        "bddl_file_name": task.bddl_file,
        "camera_heights": camera_height,
        "camera_widths": camera_width,
        "camera_names": ["agentview", "robot0_eye_in_hand"],
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    env.reset()
    return env, task


def rollout_episode(server, env, task_desc: str, max_steps: int = 600) -> bool:
    """Run a single evaluation episode. Returns True if successful."""
    obs = env.reset()
    server._reset(prompt=task_desc)

    # Initial observation
    init_obs = format_obs(obs, is_first=True)
    server.infer({"reset": True, "prompt": task_desc})

    step = 0
    done = False
    success = False

    while step < max_steps and not done:
        formatted = format_obs(obs, is_first=(step == 0))

        if step == 0:
            result = server.infer(formatted)
        else:
            server.infer({**formatted, "compute_kv_cache": True})
            result = server.infer(formatted)

        if "action" not in result:
            step += 1
            continue

        actions = result["action"]
        # Execute actions in environment
        for a_idx in range(actions.shape[0]):
            action = actions[a_idx]
            obs, reward, done, info = env.step(action)
            step += 1
            if done:
                success = info.get("success", reward > 0)
                break

    return success


def format_obs(obs: dict, is_first: bool = False) -> dict:
    """Convert LIBERO env observation to VA_Server format."""
    images = {
        "agentview_image": obs["agentview_image"],
        "robot0_eye_in_hand_image": obs["robot0_eye_in_hand_image"],
    }

    joint_pos = obs.get("robot0_joint_pos", np.zeros(7))
    gripper = obs.get("robot0_gripper_qpos", np.zeros(2))
    state = np.concatenate([joint_pos, [gripper.mean()]])

    result = {"obs": [images]}
    if not is_first:
        state_expanded = np.tile(state, (1, 1)).astype(np.float32)
        result["state"] = state_expanded

    return result


def evaluate_suite(
    server,
    suite_name: str,
    n_eval: int,
    save_dir: Path,
    camera_h: int = 256,
    camera_w: int = 256,
) -> dict:
    """Evaluate all tasks in a suite. Returns per-task success rates."""
    tasks, bm = get_libero_tasks(suite_name)
    results = {}

    for task_idx, (task_name, task_desc) in enumerate(tasks):
        logger.info(f"  Task {task_idx}/{len(tasks)}: {task_name}")
        env, task = create_env(bm, task_idx, camera_h, camera_w)
        successes = 0

        for trial in range(n_eval):
            try:
                ok = rollout_episode(server, env, task_desc)
                if ok:
                    successes += 1
            except Exception as e:
                logger.warning(f"    Trial {trial} failed: {e}")

        sr = successes / max(1, n_eval) * 100
        results[task_name] = {"success_rate": sr, "successes": successes, "total": n_eval}
        logger.info(f"    SR = {sr:.1f}% ({successes}/{n_eval})")
        env.close()

    avg_sr = np.mean([v["success_rate"] for v in results.values()])
    results["_average"] = avg_sr
    logger.info(f"  Suite {suite_name} average SR = {avg_sr:.1f}%")

    out_path = save_dir / f"{suite_name}_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", type=str, default="libero_a2a")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--suites", nargs="+",
                        default=["libero_spatial", "libero_object", "libero_goal", "libero_long"])
    parser.add_argument("--n-eval", type=int, default=20)
    parser.add_argument("--save-dir", type=str, default="results/libero_eval")
    args = parser.parse_args()

    init_logger()
    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    init_distributed(world_size, local_rank, rank)

    config = VA_CONFIGS[args.config_name]
    config.local_rank = local_rank
    config.rank = rank
    config.world_size = world_size
    device = torch.device(f"cuda:{local_rank}")

    logger.info(f"Loading model from checkpoint: {args.checkpoint}")
    server = load_model_from_checkpoint(config, args.checkpoint, device)

    save_dir = Path(args.save_dir)
    all_results = {}

    for suite in args.suites:
        logger.info(f"=== Evaluating {suite} ===")
        results = evaluate_suite(
            server, suite, args.n_eval, save_dir,
            config.height, config.width,
        )
        all_results[suite] = results

    # Summary
    logger.info("=" * 60)
    logger.info("LIBERO Evaluation Summary")
    logger.info("=" * 60)
    suite_averages = []
    for suite, res in all_results.items():
        avg = res.get("_average", 0)
        suite_averages.append(avg)
        logger.info(f"  {suite}: {avg:.1f}%")
    logger.info(f"  Overall Average: {np.mean(suite_averages):.1f}%")

    summary_path = save_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({s: r.get("_average", 0) for s, r in all_results.items()}, f, indent=2)
    logger.info(f"Results saved to {save_dir}")


if __name__ == "__main__":
    main()
