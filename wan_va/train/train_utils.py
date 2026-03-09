# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""
Training utilities for LingBot-VA fine-tuning.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Iterator

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

def create_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float = 0.01,
    ae_lr_mult: float = 1.0,
    action_encoder: nn.Module | None = None,
    action_decoder: nn.Module | None = None,
) -> torch.optim.AdamW:
    """Build AdamW with optional separate lr for AE modules."""
    param_groups = []

    ae_params = set()
    if action_encoder is not None:
        ae_params.update(id(p) for p in action_encoder.parameters())
    if action_decoder is not None:
        ae_params.update(id(p) for p in action_decoder.parameters())

    main_params = [p for p in model.parameters() if p.requires_grad and id(p) not in ae_params]
    ae_param_list = [p for p in model.parameters() if p.requires_grad and id(p) in ae_params]

    if main_params:
        param_groups.append({"params": main_params, "lr": lr, "weight_decay": weight_decay})
    if ae_param_list:
        param_groups.append({
            "params": ae_param_list,
            "lr": lr * ae_lr_mult,
            "weight_decay": weight_decay,
        })

    # Fallback if nothing collected (e.g. FSDP flattened params)
    if not param_groups:
        param_groups = [{"params": list(model.parameters()), "lr": lr, "weight_decay": weight_decay}]

    return torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)


# ---------------------------------------------------------------------------
# LR Scheduler (cosine with linear warmup)
# ---------------------------------------------------------------------------

class CosineWarmupScheduler(torch.optim.lr_scheduler.LambdaLR):

    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int):
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        super().__init__(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Timestep sampling with BSMNTW weighting
# ---------------------------------------------------------------------------

def sample_timestep_weighted(
    scheduler,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Sample training timesteps using the scheduler's BSMNTW weighting.

    Returns integer timestep indices of shape ``(batch_size,)``.
    """
    if hasattr(scheduler, "linear_timesteps_weights") and scheduler.training:
        weights = scheduler.linear_timesteps_weights
        indices = torch.multinomial(weights, batch_size, replacement=True)
        return scheduler.timesteps[indices].to(device)
    n = len(scheduler.timesteps)
    indices = torch.randint(0, n, (batch_size,))
    return scheduler.timesteps[indices].to(device)


# ---------------------------------------------------------------------------
# Checkpointing (FSDP-aware)
# ---------------------------------------------------------------------------

def save_checkpoint(
    transformer: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    output_dir: str | Path,
    action_encoder: nn.Module | None = None,
    action_decoder: nn.Module | None = None,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rank = dist.get_rank() if dist.is_initialized() else 0

    ckpt_dir = output_dir / f"step_{step:06d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save FSDP full state dict on rank 0
    if isinstance(transformer, FSDP):
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(transformer, StateDictType.FULL_STATE_DICT, cfg):
            state = transformer.state_dict()
        if rank == 0:
            torch.save(state, ckpt_dir / "transformer.pt")
    else:
        if rank == 0:
            torch.save(transformer.state_dict(), ckpt_dir / "transformer.pt")

    if rank == 0:
        if action_encoder is not None:
            torch.save(action_encoder.state_dict(), ckpt_dir / "action_encoder.pt")
        if action_decoder is not None:
            torch.save(action_decoder.state_dict(), ckpt_dir / "action_decoder.pt")
        torch.save({"step": step}, ckpt_dir / "meta.pt")
        # Optimizer state is large; save only periodically if needed
        torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")

    if dist.is_initialized():
        dist.barrier()

    if rank == 0:
        print(f"[CKPT] Saved checkpoint at step {step} → {ckpt_dir}")


def load_checkpoint(
    ckpt_dir: str | Path,
    transformer: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    action_encoder: nn.Module | None = None,
    action_decoder: nn.Module | None = None,
) -> int:
    """Load checkpoint and return the training step."""
    ckpt_dir = Path(ckpt_dir)
    meta = torch.load(ckpt_dir / "meta.pt", map_location="cpu", weights_only=True)
    step = meta["step"]

    # Transformer
    tf_state = torch.load(ckpt_dir / "transformer.pt", map_location="cpu", weights_only=True)
    if isinstance(transformer, FSDP):
        with FSDP.state_dict_type(transformer, StateDictType.FULL_STATE_DICT):
            transformer.load_state_dict(tf_state)
    else:
        transformer.load_state_dict(tf_state, strict=False)

    if action_encoder is not None and (ckpt_dir / "action_encoder.pt").exists():
        action_encoder.load_state_dict(
            torch.load(ckpt_dir / "action_encoder.pt", map_location="cpu", weights_only=True)
        )
    if action_decoder is not None and (ckpt_dir / "action_decoder.pt").exists():
        action_decoder.load_state_dict(
            torch.load(ckpt_dir / "action_decoder.pt", map_location="cpu", weights_only=True)
        )
    if optimizer is not None and (ckpt_dir / "optimizer.pt").exists():
        optimizer.load_state_dict(
            torch.load(ckpt_dir / "optimizer.pt", map_location="cpu", weights_only=True)
        )
    print(f"[CKPT] Resumed from step {step} ← {ckpt_dir}")
    return step


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def setup_training_fsdp(
    transformer: nn.Module,
    device_id: int,
    param_dtype: torch.dtype = torch.bfloat16,
) -> nn.Module:
    """Wrap transformer with FSDP for *training* (requires_grad=True)."""
    from functools import partial
    from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
    from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy

    transformer.train().requires_grad_(True)

    if dist.is_initialized():
        transformer = FSDP(
            module=transformer,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            auto_wrap_policy=partial(
                lambda_auto_wrap_policy,
                lambda_fn=lambda m: m in transformer.blocks,
            ),
            mixed_precision=MixedPrecision(
                param_dtype=param_dtype,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.float32,
            ),
            device_id=device_id,
            sync_module_states=True,
            use_orig_params=True,
        )
    else:
        transformer.to(param_dtype).to(torch.device(f"cuda:{device_id}"))

    return transformer
