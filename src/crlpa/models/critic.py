from __future__ import annotations

import torch
from torch import nn

from crlpa.models.actor import mlp


class ValueCritic(nn.Module):
    """Estimates the expected discounted reward-to-go V(s)."""

    def __init__(self, obs_dim: int, hidden: tuple[int, ...] = (64, 64)) -> None:
        super().__init__()
        self.net = mlp(obs_dim, hidden, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)
