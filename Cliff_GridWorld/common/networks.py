from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


def build_mlp(input_dim: int, hidden_dims: Iterable[int], output_dim: int) -> nn.Sequential:
    layers = []
    prev = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU())
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


class CategoricalActor(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden_dims: tuple[int, ...] = (64, 64)) -> None:
        super().__init__()
        self.net = build_mlp(input_dim, hidden_dims, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...] = (64, 64)) -> None:
        super().__init__()
        self.net = build_mlp(input_dim, hidden_dims, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SharedActorCritic(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden_dims: tuple[int, ...] = (64, 64)) -> None:
        super().__init__()
        if len(hidden_dims) == 0:
            raise ValueError("hidden_dims must contain at least one layer.")

        body_layers = []
        prev = input_dim
        for h in hidden_dims:
            body_layers.append(nn.Linear(prev, h))
            body_layers.append(nn.ReLU())
            prev = h
        self.body = nn.Sequential(*body_layers)
        self.policy_head = nn.Linear(prev, action_dim)
        self.value_head = nn.Linear(prev, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.body(x)
        logits = self.policy_head(hidden)
        value = self.value_head(hidden)
        return logits, value

