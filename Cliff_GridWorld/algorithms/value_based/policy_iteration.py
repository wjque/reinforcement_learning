from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import random
from tqdm import tqdm

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
        for i in tqdm(range(self.max_iterations)):
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
                new_v = self._bellman_expectation_backup(env, state)
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
            best_action = self._greedy_action(env, state)
            self.policy[state] = 0.0
            self.policy[state, best_action] = 1.0
            if best_action != old_action:
                stable = False
        return stable

    def _bellman_expectation_backup(self, env: Any, state: int) -> float:
        # Bellman expectation backup for policy evaluation.
        # V(s) = sum_a pi(a|s) * [r(s,a,s') + gamma * V(s')]
        new_v = 0
        for action in self.action_space:
            next_state, reward, done, info = env.transition(state, action)
            if done:
                new_v += self.policy[state][action] * reward
            else:
                new_v += self.policy[state][action] * (reward + self.gamma * self.value[next_state])
        return new_v

    def _greedy_action(self, env: Any, state: int) -> int:
        # Policy improvement step.
        # Pick argmax_a [r(s,a,s') + gamma * V(s')].
        if random.random() < 0.1:
            return random.choice(self.action_space)
        else:
            q_map = []
            for action in self.action_space:
                next_state, reward, done, info = env.transition(state, action)
                if done:
                    q_map.append(reward)
                else:
                    q_map.append(reward + self.gamma * self.value[next_state])
            return int(np.argmax(q_map))

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

