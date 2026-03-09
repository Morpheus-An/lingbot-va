# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""
A2A / F2F flow matching training utilities.

These functions are designed to be integrated into an external training
pipeline. They implement the three loss terms from the A2A paper:

  L_total = lambda1 * L_FM  +  lambda2 * L_AE  +  lambda3 * L_IC

where:
  L_FM  = flow matching loss (velocity field regression)
  L_AE  = autoencoder reconstruction loss
  L_IC  = inference consistency loss (ODE output vs ground truth)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from wan_va.utils.scheduler import FlowMatchScheduler


def compute_a2a_flow_loss(
    model_output: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    A2A flow matching loss (Eq. 3 in paper).

    Standard FM: loss = ||f_theta(x_sigma, sigma, c) - (noise - target)||^2
    A2A FM:      loss = ||f_theta(x_sigma, sigma, c) - (source - target)||^2
    """
    velocity_target = source - target
    return F.mse_loss(model_output, velocity_target)


def compute_ae_reconstruction_loss(
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    actions: torch.Tensor,
) -> torch.Tensor:
    """
    Autoencoder reconstruction loss (Eq. 4 in paper).
    L_AE = ||actions - Decoder(Encoder(actions))||_1
    """
    return F.l1_loss(decoder(encoder(actions)), actions)


def compute_consistency_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    decoder: torch.nn.Module | None = None,
    target_actions: torch.Tensor | None = None,
    lambda_decode: float = 0.5,
) -> torch.Tensor:
    """
    Inference consistency loss (Eq. 5 in paper).
    L_IC = ||z1_pred - Encoder(target)||_1  +  lambda_0 * ||Decoder(z1_pred) - target||_1

    Args:
        predicted: ODE-integrated latent z1_pred
        target: Encoder(ground_truth_actions)
        decoder: ActionDecoder (optional, for decoded-space consistency)
        target_actions: raw ground truth actions (needed when decoder is given)
        lambda_decode: weight for decoded-space term
    """
    loss = F.l1_loss(predicted, target)
    if decoder is not None and target_actions is not None:
        loss = loss + lambda_decode * F.l1_loss(decoder(predicted), target_actions)
    return loss


def prepare_a2a_training_sample(
    scheduler: FlowMatchScheduler,
    target: torch.Tensor,
    source: torch.Tensor,
    timestep: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare a noisy sample and velocity target for A2A training.

    Returns:
        noisy_sample: (1-sigma)*target + sigma*source
        velocity_target: source - target
    """
    noisy_sample = scheduler.add_noise_from_source(target, source, timestep)
    velocity_target = scheduler.training_target_from_source(target, source, timestep)
    return noisy_sample, velocity_target
