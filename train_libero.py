#!/usr/bin/env python3
# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""
Fine-tune LingBot-VA (with A2A/F2F flow matching) on LIBERO.

Usage:
    torchrun --nproc_per_node=4 train_libero.py \
        --config-name libero_a2a \
        --data-root data/libero_preprocessed \
        --output-dir checkpoints/libero-a2a-ft
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import DataLoader, DistributedSampler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "wan_va"))

from wan_va.configs import VA_CONFIGS
from wan_va.data.libero_dataset import LIBERODataset, collate_libero
from wan_va.distributed.util import init_distributed
from wan_va.modules.utils import load_text_encoder, load_tokenizer, load_transformer
from wan_va.train.train_utils import (
    CosineWarmupScheduler,
    create_optimizer,
    is_main_process,
    save_checkpoint,
    load_checkpoint,
    sample_timestep_weighted,
    setup_training_fsdp,
)
from wan_va.utils import (
    FlowMatchScheduler,
    data_seq_to_patch,
    get_mesh_id,
    init_logger,
    logger,
)
from wan_va.utils.a2a_training import (
    compute_a2a_flow_loss,
    compute_ae_reconstruction_loss,
    compute_consistency_loss,
)


# ---------------------------------------------------------------------------
# Prompt encoding (reuse VA_Server logic but standalone)
# ---------------------------------------------------------------------------

def encode_prompt_batch(
    prompts: list[str],
    tokenizer,
    text_encoder,
    device: torch.device,
    dtype: torch.dtype,
    max_seq_len: int = 512,
    cfg_dropout_rate: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Encode a batch of text prompts. Returns (prompt_embeds, negative_embeds)."""
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    prompts = [prompt_clean(p) for p in prompts]

    inputs = tokenizer(
        prompts, padding="max_length", max_length=max_seq_len,
        truncation=True, add_special_tokens=True,
        return_attention_mask=True, return_tensors="pt",
    )
    ids, mask = inputs.input_ids.to(device), inputs.attention_mask.to(device)
    seq_lens = mask.gt(0).sum(dim=1).long()

    with torch.no_grad():
        embeds = text_encoder(ids, mask).last_hidden_state.to(dtype)

    embeds = [u[:v] for u, v in zip(embeds, seq_lens)]
    embeds = torch.stack([
        torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))])
        for u in embeds
    ])

    neg_embeds = None
    if cfg_dropout_rate > 0:
        neg_inputs = tokenizer(
            [""] * len(prompts), padding="max_length", max_length=max_seq_len,
            truncation=True, add_special_tokens=True,
            return_attention_mask=True, return_tensors="pt",
        )
        neg_ids = neg_inputs.input_ids.to(device)
        neg_mask = neg_inputs.attention_mask.to(device)
        with torch.no_grad():
            neg_embeds = text_encoder(neg_ids, neg_mask).last_hidden_state.to(dtype)
        neg_seq = neg_mask.gt(0).sum(dim=1).long()
        neg_embeds = [u[:v] for u, v in zip(neg_embeds, neg_seq)]
        neg_embeds = torch.stack([
            torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))])
            for u in neg_embeds
        ])

    return embeds, neg_embeds


# ---------------------------------------------------------------------------
# Single training step
# ---------------------------------------------------------------------------

def train_step(
    batch: dict,
    transformer: torch.nn.Module,
    scheduler: FlowMatchScheduler,
    action_scheduler: FlowMatchScheduler,
    prompt_embeds: torch.Tensor,
    config,
    action_encoder: torch.nn.Module | None = None,
    action_decoder: torch.nn.Module | None = None,
    action_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Execute one training step; returns a dict of loss tensors."""
    device = prompt_embeds.device
    dtype = prompt_embeds.dtype
    fcs = config.frame_chunk_size
    patch_size = config.patch_size
    use_a2a = getattr(config, "use_a2a", False)
    use_f2f = getattr(config, "use_f2f", False)

    target_latents = batch["target_latents"].to(device, dtype)  # (B, 48, F, H, W)
    target_actions = batch["target_actions"].to(device, dtype)  # (B, 30, F, APF, 1)
    B = target_latents.shape[0]
    lat_F = target_latents.shape[2]
    lat_H, lat_W = target_latents.shape[3], target_latents.shape[4]

    losses = {}

    # ---- Video flow matching loss ----
    v_timestep = sample_timestep_weighted(scheduler, B, device)

    if use_f2f and batch["f2f_source"] is not None:
        f2f_source = batch["f2f_source"].to(device, dtype)
        # Ensure same spatial/temporal dims
        if f2f_source.shape[2:] != target_latents.shape[2:]:
            f2f_source = F.interpolate(
                f2f_source.flatten(0, 1), size=target_latents.shape[3:], mode="bilinear"
            ).view(B, 48, -1, lat_H, lat_W)
            f2f_source = f2f_source[:, :, :lat_F]
        noisy_video = scheduler.add_noise_from_source(target_latents, f2f_source, v_timestep)
        video_velocity_target = f2f_source - target_latents
    else:
        noise_v = torch.randn_like(target_latents)
        noisy_video = scheduler.add_noise(target_latents, noise_v, v_timestep)
        video_velocity_target = noise_v - target_latents

    # Build video input_dict
    frame_st_id = batch["frame_st_id"][0].item()
    v_grid = get_mesh_id(
        noisy_video.shape[2] // patch_size[0],
        lat_H // patch_size[1],
        lat_W // patch_size[2],
        0, 1, frame_st_id,
    ).to(device)
    v_timestep_expanded = torch.ones(
        noisy_video.shape[2], dtype=torch.float32, device=device
    ) * v_timestep[0]

    v_input = {
        "noisy_latents": noisy_video,
        "timesteps": v_timestep_expanded[None].expand(B, -1),
        "grid_id": v_grid[None].expand(B, -1, -1),
        "text_emb": prompt_embeds[:B],
    }

    video_pred = transformer(v_input, update_cache=0, action_mode=False)
    video_pred = data_seq_to_patch(
        patch_size, video_pred, lat_F, lat_H, lat_W, batch_size=B,
    )
    losses["video_fm"] = F.mse_loss(video_pred, video_velocity_target)

    # ---- Action flow matching loss ----
    a_timestep = sample_timestep_weighted(action_scheduler, B, device)

    clean_actions = target_actions.clone()
    if action_mask is not None:
        clean_actions[:, ~action_mask] = 0.0

    if use_a2a and batch["a2a_source"] is not None and action_encoder is not None:
        a2a_source = batch["a2a_source"].to(device, dtype)
        encoded_source = action_encoder(a2a_source)
        encoded_target = action_encoder(clean_actions)
        noisy_actions = action_scheduler.add_noise_from_source(
            encoded_target, encoded_source, a_timestep
        )
        action_velocity_target = encoded_source - encoded_target
    else:
        noise_a = torch.randn_like(clean_actions)
        noisy_actions = action_scheduler.add_noise(clean_actions, noise_a, a_timestep)
        action_velocity_target = noise_a - clean_actions

    apf = config.action_per_frame
    a_grid = get_mesh_id(
        noisy_actions.shape[2], apf, 1, 1, 1, frame_st_id, action=True,
    ).to(device)
    a_timestep_expanded = torch.ones(
        noisy_actions.shape[2], dtype=torch.float32, device=device,
    ) * a_timestep[0]

    a_input = {
        "noisy_latents": noisy_actions,
        "timesteps": a_timestep_expanded[None].expand(B, -1),
        "grid_id": a_grid[None].expand(B, -1, -1),
        "text_emb": prompt_embeds[:B],
    }

    action_pred_seq = transformer(a_input, update_cache=0, action_mode=True)
    action_pred = rearrange(action_pred_seq, "b (f n) c -> b c f n 1", f=fcs)
    losses["action_fm"] = F.mse_loss(action_pred, action_velocity_target)

    # ---- AE reconstruction loss (A2A only) ----
    if use_a2a and action_encoder is not None and action_decoder is not None:
        losses["ae_recon"] = compute_ae_reconstruction_loss(
            action_encoder, action_decoder, clean_actions,
        )

        # ActionDecoder pass on the flow-matched output for consistency
        # ODE integration is expensive; approximate with single-step prediction
        with torch.no_grad():
            a_tid = torch.argmin(
                (action_scheduler.timesteps[:, None].to(device) - a_timestep[None]).abs(), dim=0
            )
            sigma_a = action_scheduler.sigmas[a_tid[0]].to(noisy_actions)
            z1_approx = noisy_actions + action_pred * (0 - sigma_a)
        losses["consistency"] = compute_consistency_loss(
            z1_approx,
            encoded_target if use_a2a else clean_actions,
            decoder=action_decoder,
            target_actions=clean_actions,
            lambda_decode=config.lambda_decode,
        )

    return losses


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", type=str, default="libero_a2a")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="checkpoints/libero-a2a-ft")
    parser.add_argument("--train-steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint dir to resume from")
    parser.add_argument("--suites", nargs="+", default=None)
    args = parser.parse_args()

    # ---- distributed setup ----
    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    init_distributed(world_size, local_rank, rank)
    device = torch.device(f"cuda:{local_rank}")
    init_logger()

    config = VA_CONFIGS[args.config_name]
    train_steps = args.train_steps or config.train_steps
    lr = args.lr or config.lr
    dtype = config.param_dtype

    if is_main_process():
        logger.info(f"Config: {config.__name__}")
        logger.info(f"Train steps: {train_steps}, LR: {lr}, Batch: {args.batch_size}")
        logger.info(f"A2A={config.use_a2a}, F2F={config.use_f2f}")

    # ---- Load models ----
    model_path = os.path.expanduser(config.wan22_pretrained_model_name_or_path)

    if is_main_process():
        logger.info(f"Loading tokenizer & text encoder from {model_path}")
    tokenizer = load_tokenizer(os.path.join(model_path, "tokenizer"))
    text_encoder = load_text_encoder(
        os.path.join(model_path, "text_encoder"), dtype, device,
    )
    text_encoder.eval().requires_grad_(False)

    if is_main_process():
        logger.info("Loading transformer...")
    transformer = load_transformer(
        os.path.join(model_path, "transformer"), dtype, torch.device("cpu"),
    )
    transformer = setup_training_fsdp(transformer, local_rank, dtype)
    if is_main_process():
        logger.info(f"Transformer ready (FSDP={dist.is_initialized()})")

    # A2A modules
    action_encoder = None
    action_decoder = None
    if getattr(config, "use_a2a", False):
        from wan_va.modules.a2a_modules import ActionEncoder, ActionDecoder
        action_encoder = ActionEncoder(config.action_dim).to(device, dtype)
        action_decoder = ActionDecoder(config.action_dim).to(device, dtype)
        action_encoder.train().requires_grad_(True)
        action_decoder.train().requires_grad_(True)
        if is_main_process():
            ae_p = sum(p.numel() for p in action_encoder.parameters()) / 1e6
            ad_p = sum(p.numel() for p in action_decoder.parameters()) / 1e6
            logger.info(f"ActionEncoder ({ae_p:.3f}M) + ActionDecoder ({ad_p:.3f}M) initialized")

    # Action mask
    action_mask = torch.zeros(config.action_dim, dtype=torch.bool, device=device)
    action_mask[config.used_action_channel_ids] = True

    # ---- Schedulers ----
    scheduler = FlowMatchScheduler(shift=config.snr_shift, sigma_min=0.0, extra_one_step=True)
    action_scheduler = FlowMatchScheduler(shift=config.action_snr_shift, sigma_min=0.0, extra_one_step=True)
    scheduler.set_timesteps(1000, training=True)
    action_scheduler.set_timesteps(1000, training=True)

    # ---- Load real norm stats (override config placeholders) ----
    norm_stats_path = os.path.join(args.data_root, "norm_stats_config.json")
    if os.path.isfile(norm_stats_path):
        from wan_va.data.normalize import load_norm_stats
        config.norm_stat = load_norm_stats(norm_stats_path)
        if is_main_process():
            logger.info(f"Loaded norm stats from {norm_stats_path}")
    else:
        if is_main_process():
            logger.warning(f"No norm_stats_config.json found at {norm_stats_path}, using config defaults")

    # ---- Dataset & DataLoader ----
    dataset = LIBERODataset(args.data_root, config, suites=args.suites)
    if is_main_process():
        logger.info(f"Dataset: {len(dataset)} chunks from {args.data_root}")

    sampler = DistributedSampler(dataset, shuffle=True) if dist.is_initialized() else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_libero,
        drop_last=True,
    )

    # ---- Optimizer & Scheduler ----
    all_params = list(transformer.parameters())
    if action_encoder is not None:
        all_params += list(action_encoder.parameters())
    if action_decoder is not None:
        all_params += list(action_decoder.parameters())

    # Wrap into a single param group for FSDP compatibility
    optimizer = torch.optim.AdamW(
        [p for p in all_params if p.requires_grad],
        lr=lr, weight_decay=config.weight_decay, betas=(0.9, 0.999),
    )
    lr_scheduler = CosineWarmupScheduler(optimizer, config.warmup_steps, train_steps)

    # ---- Resume ----
    start_step = 0
    if args.resume:
        start_step = load_checkpoint(
            args.resume, transformer, optimizer, action_encoder, action_decoder,
        )

    # ---- Training loop ----
    data_iter = iter(loader)
    global_step = start_step
    optimizer.zero_grad()
    t_start = time.time()

    while global_step < train_steps:
        # Get batch (cycle through dataset)
        try:
            batch = next(data_iter)
        except StopIteration:
            if sampler is not None:
                sampler.set_epoch(global_step)
            data_iter = iter(loader)
            batch = next(data_iter)

        # Encode prompts with CFG dropout
        prompts = batch["task_description"]
        if config.cfg_dropout_rate > 0:
            prompts = [
                "" if torch.rand(1).item() < config.cfg_dropout_rate else p
                for p in prompts
            ]
        prompt_embeds, _ = encode_prompt_batch(
            prompts, tokenizer, text_encoder, device, dtype,
        )

        with torch.amp.autocast("cuda", dtype=dtype):
            losses = train_step(
                batch, transformer, scheduler, action_scheduler,
                prompt_embeds, config,
                action_encoder, action_decoder, action_mask,
            )

        # Aggregate total loss
        total_loss = losses["video_fm"] + losses["action_fm"]
        if "ae_recon" in losses:
            total_loss = total_loss + config.lambda_ae * losses["ae_recon"]
        if "consistency" in losses:
            total_loss = total_loss + config.lambda_ic * losses["consistency"]

        scaled_loss = total_loss / args.grad_accum_steps
        scaled_loss.backward()

        if (global_step + 1) % args.grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in all_params if p.requires_grad],
                config.grad_clip_norm,
            )
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        global_step += 1

        # Logging
        if is_main_process() and global_step % config.log_every == 0:
            elapsed = time.time() - t_start
            loss_str = " | ".join(f"{k}={v.item():.4f}" for k, v in losses.items())
            cur_lr = optimizer.param_groups[0]["lr"]
            logger.info(
                f"[Step {global_step}/{train_steps}] "
                f"loss={total_loss.item():.4f} ({loss_str}) "
                f"lr={cur_lr:.2e} "
                f"elapsed={elapsed:.0f}s"
            )

        # Save checkpoint
        if global_step % config.save_every == 0:
            save_checkpoint(
                transformer, optimizer, global_step, args.output_dir,
                action_encoder, action_decoder,
            )

    # Final save
    save_checkpoint(
        transformer, optimizer, global_step, args.output_dir,
        action_encoder, action_decoder,
    )
    if is_main_process():
        logger.info(f"Training complete. {global_step} steps in {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
