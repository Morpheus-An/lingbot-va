# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""
A2A (Action-to-Action) Autoencoder modules for flow matching.

Shape-preserving ActionEncoder + ActionDecoder that wrap the Transformer:
  state_history -> ActionEncoder -> Transformer flow matching -> ActionDecoder -> future_actions

Both use residual connections with zero-initialized final layers so that
before fine-tuning they act as identity mappings.
"""
import torch
import torch.nn as nn


class ActionEncoder(nn.Module):
    """
    Encodes proprioceptive state history into the flow matching source distribution.
    Wraps the Transformer input side.

    Input/Output shape: [B, action_dim, frame_chunk_size, action_per_frame, 1]
    """

    def __init__(self, action_dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or action_dim * 2
        self.net = nn.Sequential(
            nn.Conv3d(action_dim, hidden_dim, kernel_size=(1, 3, 1), padding=(0, 1, 0)),
            nn.SiLU(),
            nn.Conv3d(hidden_dim, action_dim, kernel_size=(1, 3, 1), padding=(0, 1, 0)),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ActionDecoder(nn.Module):
    """
    Decodes Transformer flow matching output back into real action space.
    Wraps the Transformer output side.

    Input/Output shape: [B, action_dim, frame_chunk_size, action_per_frame, 1]
    """

    def __init__(self, action_dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or action_dim * 2
        self.net = nn.Sequential(
            nn.Conv3d(action_dim, hidden_dim, kernel_size=(1, 3, 1), padding=(0, 1, 0)),
            nn.SiLU(),
            nn.Conv3d(hidden_dim, action_dim, kernel_size=(1, 3, 1), padding=(0, 1, 0)),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)
