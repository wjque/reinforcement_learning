from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np

from common.base_agent import BaseAgent


class PolicyIterationAgent(BaseAgent):
    algorithm_name = "policy_iteration"
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
        self.policy = np.full((state_space, action_space), 1.0 / action_space, dtype=np.float64)

    def train(self, env: Any, episodes: int = 0, seed: int | None = None) -> Dict[str, Any]:
        del episodes, seed
        for i in range(self.max_iterations):
            self._policy_evaluation(env)
            stable = self._policy_improvement(env)
            if stable:
                return {"iterations": i + 1, "converged": True}
        return {"iterations": self.max_iterations, "converged": False}

    def _policy_evaluation(self, env: Any) -> None:
        while True:
            delta = 0.0
            for state in range(self.state_space):
                if env.is_terminal(state):
                    continue
                old_v = self.value[state]
                new_v = self._bellman_expectation_backup_todo(env, state)
                self.value[state] = new_v
                delta = max(delta, abs(old_v - new_v))
            if delta < self.theta:
                break

    def _policy_improvement(self, env: Any) -> bool:
        stable = True
        for state in range(self.state_space):
            if env.is_terminal(state):
                continue
            old_action = int(np.argmax(self.policy[state]))
            best_action = self._greedy_action_todo(env, state)
            self.policy[state] = 0.0
            self.policy[state, best_action] = 1.0
            if best_action != old_action:
                stable = False
        return stable

    def _bellman_expectation_backup_todo(self, env: Any, state: int) -> float:
        # TODO(student): Bellman expectation backup for policy evaluation.
        # Formula target:
        # V(s) = sum_a pi(a|s) * [r(s,a,s') + gamma * V(s')]
        raise NotImplementedError("TODO(student): implement Bellman expectation backup.")

    def _greedy_action_todo(self, env: Any, state: int) -> int:
        # TODO(student): Policy improvement step.
        # Pick argmax_a [r(s,a,s') + gamma * V(s')].
        raise NotImplementedError("TODO(student): implement greedy action extraction.")

    def act(self, state: int, deterministic: bool = True) -> int:
        if deterministic:
            return int(np.argmax(self.policy[state]))
        return int(np.random.choice(self.action_space, p=self.policy[state]))

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, value=self.value, policy=self.policy)

    def load(self, path: str) -> None:
        data = np.load(path)
        self.value = data["value"]
        self.policy = data["policy"]

    def get_policy(self) -> np.ndarray:
        return np.argmax(self.policy, axis=1).astype(np.int64)

