from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


def build_mlp(input_dim: int, hidden_dims: Iterable[int], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = input_dim
    for hidden in hidden_dims:
        layers.append(nn.Linear(prev, hidden))
        layers.append(nn.ReLU())
        prev = hidden
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


class SharedPolicyValueNet(nn.Module):
    """Shared trunk with policy and value heads."""

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__()
        if len(hidden_dims) == 0:
            raise ValueError("hidden_dims must contain at least one layer.")

        trunk_layers: list[nn.Module] = []
        prev = input_dim
        for hidden in hidden_dims:
            trunk_layers.append(nn.Linear(prev, hidden))
            trunk_layers.append(nn.ReLU())
            prev = hidden

        self.trunk = nn.Sequential(*trunk_layers)
        self.policy_head = nn.Linear(prev, action_dim)
        self.value_head = nn.Linear(prev, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(x)
        logits = self.policy_head(hidden)
        value = self.value_head(hidden)
        return logits, value

