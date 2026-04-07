from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np


class BaseAgent(ABC):
    """Common interface used by train.py and play.py."""

    algorithm_name: str = "base"
    checkpoint_type: str = "npz"

    def __init__(self, state_space: int, action_space: int) -> None:
        self.state_space = state_space
        self.action_space = action_space

    @abstractmethod
    def train(self, env: Any, episodes: int = 500, seed: int | None = None) -> Dict[str, Any]:
        pass

    @abstractmethod
    def act(self, state: int, deterministic: bool = True) -> int:
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        pass

    @abstractmethod
    def get_policy(self) -> np.ndarray:
        """Return shape [state_space] policy as action indices."""
        pass

