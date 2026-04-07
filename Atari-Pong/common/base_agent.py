from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np


class BaseAgent(ABC):
    """Shared agent interface for Atari-Pong training and play."""

    algorithm_name: str = "base"
    checkpoint_type: str = "pt"

    def __init__(self, obs_dim: int, action_space: int) -> None:
        self.obs_dim = obs_dim
        self.action_space = action_space

    @abstractmethod
    def train(
        self,
        env: Any,
        total_steps: int = 300_000,
        seed: int | None = None,
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    def act(self, obs: np.ndarray, deterministic: bool = True) -> int:
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        pass

