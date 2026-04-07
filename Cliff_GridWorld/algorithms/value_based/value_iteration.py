from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np

from common.base_agent import BaseAgent


class ValueIterationAgent(BaseAgent):
    algorithm_name = "value_iteration"
    checkpoint_type = "npz"

    def __init__(
        self,
        state_space: int,
        action_space: int,
        gamma: float = 0.99,
        theta: float = 1e-6,
        max_iterations: int = 1000,
    ) -> None:
        super().__init__(state_space, action_space)
        self.gamma = gamma
        self.theta = theta
        self.max_iterations = max_iterations

        self.value = np.zeros(state_space, dtype=np.float64)
        self.policy = np.zeros(state_space, dtype=np.int64)

    def train(self, env: Any, episodes: int = 0, seed: int | None = None) -> Dict[str, Any]:
        del episodes, seed
        converged = False
        iters = 0
        for i in range(self.max_iterations):
            delta = 0.0
            for state in range(self.state_space):
                if env.is_terminal(state):
                    continue
                old_v = self.value[state]
                self.value[state] = self._optimality_backup(env, state)
                delta = max(delta, abs(old_v - self.value[state]))
            iters = i + 1
            if delta < self.theta:
                converged = True
                break

        for state in range(self.state_space):
            if env.is_terminal(state):
                self.policy[state] = 0
            else:
                self.policy[state] = self._extract_policy_action(env, state)

        return {"iterations": iters, "converged": converged}

    def _optimality_backup(self, env: Any, state: int) -> float:
        # Bellman optimality backup.
        # V(s) = max_a [r(s,a,s') + gamma * V(s')]
        new_vs = np.zeros(self.action_space, dtype=np.float64)
        for action in range(self.action_space):
            next_state, reward, done, info = env.transition(state, action)
            new_vs[action] = reward
            if not done:
                new_vs[action] += self.gamma * self.value[next_state]
        return np.max(new_vs)

    def _extract_policy_action(self, env: Any, state: int) -> int:
        # Policy extraction after value iteration.
        # Pick argmax_a [r(s,a,s') + gamma * V(s')].
        raise NotImplementedError("TODO: implement greedy policy extraction.")

    def act(self, state: int, deterministic: bool = True) -> int:
        if deterministic:
            return int(self.policy[state])
        return int(np.random.randint(0, self.action_space))

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, value=self.value, policy=self.policy)

    def load(self, path: str) -> None:
        data = np.load(path)
        self.value = data["value"]
        self.policy = data["policy"]

    def get_policy(self) -> np.ndarray:
        return self.policy.copy()

