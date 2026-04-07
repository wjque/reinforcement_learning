from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np

from common.base_agent import BaseAgent


class QLearningAgent(BaseAgent):
    algorithm_name = "q_learning"
    checkpoint_type = "npz"

    def __init__(
        self,
        state_space: int,
        action_space: int,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.2,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
    ) -> None:
        super().__init__(state_space, action_space)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.q_table = np.zeros((state_space, action_space), dtype=np.float64)
        self._rng = np.random.default_rng()

    def train(self, env: Any, episodes: int = 500, seed: int | None = None) -> Dict[str, Any]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        returns: list[float] = []
        for ep in range(episodes):
            state = env.reset(seed=None if seed is None else seed + ep)
            episode_return = 0.0

            for _ in range(env.max_steps):
                action = self._epsilon_greedy(state)
                next_state, reward, done, _ = env.step(action)
                episode_return += reward

                self._q_learning_update_todo(state, action, reward, next_state, done)

                state = next_state
                if done:
                    break

            returns.append(episode_return)
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return {"episode_returns": returns}

    def _epsilon_greedy(self, state: int) -> int:
        if self._rng.random() < self.epsilon:
            return int(self._rng.integers(0, self.action_space))
        return int(np.argmax(self.q_table[state]))

    def _q_learning_update_todo(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
    ) -> None:
        # TODO(student): Q-Learning TD update.
        # Formula target:
        # Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
        # Terminal transition should not bootstrap future value.
        raise NotImplementedError("TODO(student): implement Q-Learning update.")

    def act(self, state: int, deterministic: bool = True) -> int:
        if deterministic:
            return int(np.argmax(self.q_table[state]))
        return self._epsilon_greedy(state)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, q_table=self.q_table, epsilon=np.array([self.epsilon], dtype=np.float64))

    def load(self, path: str) -> None:
        data = np.load(path)
        self.q_table = data["q_table"]
        self.epsilon = float(data["epsilon"][0])

    def get_policy(self) -> np.ndarray:
        return np.argmax(self.q_table, axis=1).astype(np.int64)

