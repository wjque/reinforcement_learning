from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch.distributions import Categorical

from common.base_agent import BaseAgent
from common.networks import CategoricalActor, Critic

from tqdm import tqdm


class ActorCriticAgent(BaseAgent):
    algorithm_name = "actor_critic"
    checkpoint_type = "pt"

    def __init__(
        self,
        state_space: int,
        action_space: int,
        *,
        rows: int = 4,
        cols: int = 12,
        gamma: float = 0.99,
        actor_lr: float = 3e-4,
        critic_lr: float = 1e-3,
        hidden_dims: tuple[int, ...] = (64, 64),
        device: str = "cpu",
    ) -> None:
        super().__init__(state_space, action_space)
        self.rows = rows
        self.cols = cols
        self.gamma = gamma
        self.device = torch.device(device)

        self.actor = CategoricalActor(2, action_space, hidden_dims=hidden_dims).to(self.device)
        self.critic = Critic(2, hidden_dims=hidden_dims).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    def _state_to_feature(self, state: int) -> np.ndarray:
        row = state // self.cols
        col = state % self.cols
        return np.array([row / max(1, self.rows - 1), col / max(1, self.cols - 1)], dtype=np.float32)

    def _state_tensor(self, state: int) -> torch.Tensor:
        feature = self._state_to_feature(state)
        return torch.tensor(feature, dtype=torch.float32, device=self.device).unsqueeze(0)

    def train(self, env: Any, episodes: int = 500, seed: int | None = None) -> Dict[str, Any]:
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        returns: list[float] = []
        for ep in tqdm(range(episodes)):
            state = env.reset(seed=None if seed is None else seed + ep)
            done = False
            episode_return = 0.0

            states: list[np.ndarray] = []
            actions: list[int] = []
            rewards: list[float] = []
            dones: list[float] = []
            next_states: list[np.ndarray] = []

            while not done:
                state_t = self._state_tensor(state)
                logits = self.actor(state_t)
                dist = Categorical(logits=logits)
                action = int(dist.sample().item())

                next_state, reward, done, _ = env.step(action)

                states.append(self._state_to_feature(state))
                actions.append(action)
                rewards.append(float(reward))
                dones.append(float(done))
                next_states.append(self._state_to_feature(next_state))

                episode_return += reward
                state = next_state

                if len(states) >= env.max_steps:
                    break

            self._update(states, actions, rewards, dones, next_states)
            returns.append(episode_return)

        return {"episode_returns": returns}

    def _update(
        self,
        states: list[np.ndarray],
        actions: list[int],
        rewards: list[float],
        dones: list[float],
        next_states: list[np.ndarray],
    ) -> None:
        if not states:
            return

        states_t = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)
        next_states_t = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)

        values = self.critic(states_t).squeeze(-1)
        with torch.no_grad():
            next_values = self.critic(next_states_t).squeeze(-1)

        advantage = self._compute_advantage(rewards_t, dones_t, values, next_values)

        logits = self.actor(states_t)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions_t)

        actor_loss = self._actor_loss(log_probs, advantage)

        critic_loss = self._critic_loss(rewards_t, dones_t, values, next_values)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def _compute_advantage(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor,
    ) -> torch.Tensor:
        # advantage estimation for Actor-Critic.
        # Typical one-step TD advantage:
        # A_t = r_t + gamma * (1 - done_t) * V(s_{t+1}) - V(s_t)
        return rewards + self.gamma * (1 - dones) * next_values - values

    def _actor_loss(self, log_probs: torch.Tensor, advantage: torch.Tensor) -> torch.Tensor:
        # actor loss.
        # L_actor = -E[log pi(a_t|s_t) * A_t]
        all_loss = log_probs * advantage
        return all_loss.mean()

    def _critic_loss(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor,
    ) -> torch.Tensor:
        # critic loss.
        # L_critic = MSE(V(s_t), r_t + gamma * (1-done_t) * V(s_{t+1}))
        all_loss = (values - rewards - self.gamma * (1-dones) * next_values)
        return all_loss.mean()

    def act(self, state: int, deterministic: bool = True) -> int:
        with torch.no_grad():
            logits = self.actor(self._state_tensor(state))
            if deterministic:
                return int(torch.argmax(logits, dim=-1).item())
            dist = Categorical(logits=logits)
            return int(dist.sample().item())

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "rows": self.rows,
            "cols": self.cols,
            "gamma": self.gamma,
        }
        torch.save(payload, path)

    def load(self, path: str) -> None:
        payload = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(payload["actor"])
        self.critic.load_state_dict(payload["critic"])
        self.rows = int(payload.get("rows", self.rows))
        self.cols = int(payload.get("cols", self.cols))
        self.gamma = float(payload.get("gamma", self.gamma))

    def get_policy(self) -> np.ndarray:
        actions = np.zeros(self.state_space, dtype=np.int64)
        with torch.no_grad():
            for state in range(self.state_space):
                logits = self.actor(self._state_tensor(state))
                actions[state] = int(torch.argmax(logits, dim=-1).item())
        return actions

