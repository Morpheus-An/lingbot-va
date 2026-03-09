# [CLAUDE.md](http://CLAUDE.md)

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LingBot-VA is an autoregressive diffusion framework for simultaneous world modeling (video prediction) and robot action inference. It uses a shared Mixture-of-Transformers (MoT) backbone built on Wan2.2-5B, with dual streams for video and action. The system supports server-client inference via WebSocket and distributed training with FSDP.

## Requirements

- Python 3.10, PyTorch 2.9.0, CUDA 12.6
- flash-attn (install with `--no-build-isolation`)
- Key deps: diffusers==0.36.0, transformers>=4.55.2, einops, easydict

## Common Commands

## Activate Environment

```bash
conda activate lingvla
```

### Inference Server (RoboTwin)

```bash
# Single GPU
bash evaluation/robotwin/launch_server.sh
# Multi-GPU (8)
bash evaluation/robotwin/launch_server_multigpus.sh
# Image-to-Video-Action generation
NGPU=1 CONFIG_NAME='robotwin_i2av' bash script/run_launch_va_server_sync.sh
```

### Evaluation Client (RoboTwin)

```bash
# Single GPU
bash evaluation/robotwin/launch_client.sh results/ adjust_bottle
# Multi-GPU (group 0-6)
bash evaluation/robotwin/launch_client_multigpus.sh results/ 0
```

### Training (LIBERO fine-tuning)

```bash
NGPU=4 bash script/run_train_libero.sh
# Or directly:
torchrun --nproc_per_node=4 train_libero.py \
    --config-name libero_a2a \
    --data-root data/libero_preprocessed \
    --output-dir checkpoints/libero-a2a-ft
```

### Formatting

```bash
black .
isort .
```

## Architecture

### Inference Pipeline

The system uses a **server-client architecture** connected via WebSocket:

1. **VA_Server** (`wan_va/wan_va_server.py`) — Main inference entry point. Loads all components (VAE, T5, Tokenizer, Transformer), manages the autoregressive rollout loop with KV cache.
2. **WebSocket layer** (`wan_va/utils/Simple_Remote_Infer/`) — Serializes observations/actions between simulation and model.
3. **Eval client** (`evaluation/robotwin/eval_polict_client_openpi.py`) — Runs the RoboTwin simulation, sends observations, receives actions.

Launch scripts use `torch.distributed.run` for multi-GPU FSDP sharding, even for single-GPU (nproc=1).

### Core Model

`**WanTransformer3DModel`** (`wan_va/modules/model.py`):

- 30-layer Transformer with 24 attention heads, 128 dim/head (3072 inner dim)
- Shared self-attention blocks process both video latent tokens and action tokens
- Separate input/output projections: `patch_embedding_mlp` + `proj_out` for video, `action_embedder` + `action_proj_out` for actions
- Separate `condition_embedder` and `condition_embedder_action` for timestep/text conditioning
- `action_mode` flag in `forward()` switches between video and action pathways
- 3D Rotary Position Embedding (RoPE) for spatiotemporal encoding
- KV cache with slot-based allocation for efficient autoregressive inference

### Flow Matching

`**FlowMatchScheduler`** (`wan_va/utils/scheduler.py`):

- Separate schedulers for video (snr_shift=5.0) and action (snr_shift=1.0)
- Standard FM: interpolates between Gaussian noise and target
- A2A variant: `add_noise_from_source()` interpolates between source (previous action/frame) and target
- BSMNTW weighting for training timestep sampling

### A2A / F2F Flow Matching

Action-to-Action (A2A) and Frame-to-Frame (F2F) replace Gaussian noise with structured sources:

- **A2A**: Previous state history as source for action denoising (via `ActionEncoder`/`ActionDecoder` in `wan_va/modules/a2a_modules.py`)
- **F2F**: Previous video latent as source for video denoising
- Three loss terms: `L_FM` (flow matching), `L_AE` (autoencoder reconstruction), `L_IC` (inference consistency)
- Training utilities in `wan_va/utils/a2a_training.py`

### Configuration System

Configs in `wan_va/configs/` use `EasyDict`, registered in `VA_CONFIGS` dict:

- `robotwin` / `robotwin_a2a` — RoboTwin dual-arm (30D action, 3 cameras)
- `libero` / `libero_a2a` — LIBERO single-arm Franka (7D action mapped to 30D, 2 cameras)
- `robotwin_i2av` / `franka_i2av` — Image-to-video-action generation mode

Key per-config parameters: `attn_window`, `frame_chunk_size`, `action_per_frame`, `action_dim`, `num_inference_steps`, `action_num_inference_steps`, `used_action_channel_ids`, `norm_stat`.

### Action Space

All environments map to a universal **30D action space** via `used_action_channel_ids`. Actions are normalized to [-1, 1] using quantile normalization (q01/q99). The `inverse_used_action_channel_ids` mapping is used to extract active channels back.

### Distributed Inference

- FSDP wraps each Transformer block independently (`wan_va/distributed/fsdp.py`)
- Rank 0 runs the WebSocket server; other ranks run `worker_loop` waiting for broadcast commands (`wan_va/utils/sever_utils.py`)

### Training Pipeline

`train_libero.py` implements fine-tuning with:

- FSDP-wrapped transformer + optional A2A encoder/decoder
- `LIBERODataset` loads pre-encoded `.pt` episodes (output of `wan_va/data/preprocess_libero.py`)
- Gradient accumulation, cosine warmup LR schedule, FSDP-aware checkpointing
- CFG dropout on text prompts

### VAE Streaming

`WanVAEStreamingWrapper` (`wan_va/modules/utils.py`) enables chunk-by-chunk VAE encoding by maintaining causal convolution feature caches, avoiding loading all frames at once.

### Recommended skills for this project

#### Daily research essentials


| Command                                       | When to use                                                                                    |
| --------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `/superpowers:brainstorming`                  | Before designing new mechanisms — divergent thinking, generate multiple approaches then select |
| `/superpowers:systematic-debugging`           | When experiments fail or produce unexpected results — systematic root cause analysis           |
| `/simplify`                                   | After modifying code — check for redundancy, quality, and efficiency                           |
| `/superpowers:verification-before-completion` | Before committing — verify syntax, parameter chains, reset logic                               |


#### Session & progress tracking


| Command           | When to use                                                                        |
| ----------------- | ---------------------------------------------------------------------------------- |
| `/log-session`    | After important debugging or analysis sessions — save context for future reference |
| `/recall`         | Before starting related work — search what was discovered in past sessions         |
| `/daily-progress` | End of day — auto-generate progress summary from session logs + git history        |


#### Complex feature development (use when adding/refactoring planner mechanisms)


| Command                                    | When to use                                                                |
| ------------------------------------------ | -------------------------------------------------------------------------- |
| `/everything-claude-code:plan`             | Approach unclear — discuss and align on design before coding               |
| `/superpowers:writing-plans`               | Approach decided — generate detailed step-by-step implementation blueprint |
| `/planning-with-files:plan`                | Multi-day task — persistent file-based progress tracking across sessions   |
| `/everything-claude-code:python-review`    | After major changes — comprehensive Python code review                     |
| `/superpowers:dispatching-parallel-agents` | Independent tasks — e.g., analyze 3 tasks' results simultaneously          |


#### Occasionally useful


| Command                                      | When to use                                                                        |
| -------------------------------------------- | ---------------------------------------------------------------------------------- |
| `/everything-claude-code:search-first`       | Before implementing utilities — check if existing libraries/patterns exist         |
| `/superpowers:using-git-worktrees`           | Experimental branches — isolate new variants without affecting running experiments |
| `/everything-claude-code:orchestrate bugfix` | Complex bugs — full pipeline: diagnose → fix → test → review                       |
| `/everything-claude-code:strategic-compact`  | Long sessions — manually compact context at logical breakpoints                    |
| `/everything-claude-code:python-patterns`    | Unsure about Python best practices — quick reference                               |


#### Typical research workflows

```
Iterating:     brainstorming → plan → implement → simplify → verification → log-session
Big refactor:  brainstorming → writing-plans → dispatching-parallel-agents → python-review → log-session
Daily wrap-up: recall → daily-progress
```

