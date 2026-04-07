from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class A2CBuffer:
    observations: list[np.ndarray] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)
    next_observations: list[np.ndarray] = field(default_factory=list)

    def add(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        next_observation: np.ndarray,
    ) -> None:
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.next_observations.append(next_observation)

    def clear(self) -> None:
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.next_observations.clear()

    def __len__(self) -> int:
        return len(self.observations)


@dataclass
class PPOBuffer:
    observations: list[np.ndarray] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    next_observations: list[np.ndarray] = field(default_factory=list)

    def add(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
        next_observation: np.ndarray,
    ) -> None:
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.next_observations.append(next_observation)

    def clear(self) -> None:
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()
        self.next_observations.clear()

    def __len__(self) -> int:
        return len(self.observations)

