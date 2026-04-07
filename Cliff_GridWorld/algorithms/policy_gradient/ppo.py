from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm

from common.base_agent import BaseAgent
from common.buffers import RolloutBuffer
from common.networks import SharedActorCritic


class PPOAgent(BaseAgent):
    algorithm_name = "ppo"
    checkpoint_type = "pt"

    def __init__(
        self,
        state_space: int,
        action_space: int,
        *,
        rows: int = 4,
        cols: int = 12,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        learning_rate: float = 3e-4,
        ppo_epochs: int = 4,
        minibatch_size: int = 64,
        hidden_dims: tuple[int, ...] = (64, 64),
        device: str = "cpu",
    ) -> None:
        super().__init__(state_space, action_space)
        self.rows = rows
        self.cols = cols
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size
        self.device = torch.device(device)

        self.net = SharedActorCritic(2, action_space, hidden_dims=hidden_dims).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)
        self.buffer = RolloutBuffer()

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
        for ep in tqdm(range(episodes), desc=f"{episodes} Episodes"):
            state = env.reset(seed=None if seed is None else seed + ep)
            done = False
            episode_return = 0.0
            self.buffer.clear()

            while not done:
                with torch.no_grad():
                    logits, value = self.net(self._state_tensor(state))
                    dist = Categorical(logits=logits)
                    action = int(dist.sample().item())
                    log_prob = float(dist.log_prob(torch.tensor(action, device=self.device)).item())
                    state_value = float(value.squeeze(-1).item())

                next_state, reward, done, _ = env.step(action)

                self.buffer.add(
                    state=self._state_to_feature(state),
                    action=action,
                    reward=float(reward),
                    done=bool(done),
                    log_prob=log_prob,
                    value=state_value,
                    next_state=self._state_to_feature(next_state),
                )

                episode_return += reward
                state = next_state
                if len(self.buffer) >= env.max_steps:
                    break

            self._update()
            returns.append(episode_return)

        return {"episode_returns": returns}

    def _update(self) -> None:
        if len(self.buffer) == 0:
            return

        states_t = torch.tensor(np.array(self.buffer.states), dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(self.buffer.actions, dtype=torch.long, device=self.device)
        rewards_t = torch.tensor(self.buffer.rewards, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(self.buffer.dones, dtype=torch.float32, device=self.device)
        old_log_probs_t = torch.tensor(self.buffer.log_probs, dtype=torch.float32, device=self.device)
        values_t = torch.tensor(self.buffer.values, dtype=torch.float32, device=self.device)
        next_states_t = torch.tensor(np.array(self.buffer.next_states), dtype=torch.float32, device=self.device)

        with torch.no_grad():
            _, next_values_t = self.net(next_states_t)
            next_values_t = next_values_t.squeeze(-1)

        
        advantages_t, returns_t = self._compute_gae(
            rewards_t, dones_t, values_t, next_values_t
        )

        batch_size = states_t.shape[0]
        for _ in range(self.ppo_epochs):
            indices = np.random.permutation(batch_size)
            for start in range(0, batch_size, self.minibatch_size):
                mb_idx = indices[start : start + self.minibatch_size]
                mb_idx_t = torch.tensor(mb_idx, dtype=torch.long, device=self.device)

                mb_states = states_t[mb_idx_t]
                mb_actions = actions_t[mb_idx_t]
                mb_old_log_probs = old_log_probs_t[mb_idx_t]
                mb_advantages = advantages_t[mb_idx_t]
                mb_returns = returns_t[mb_idx_t]

                logits, new_values = self.net(mb_states)
                new_values = new_values.squeeze(-1)
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                policy_loss = self._policy_loss(
                    new_log_probs, mb_old_log_probs, mb_advantages
                )
                value_loss = self._value_loss(new_values, mb_returns)
                total_loss = self._total_loss(policy_loss, value_loss, entropy)

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

    def _compute_gae(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Generalized Advantage Estimation (GAE):
        # delta_t = r_t + gamma * (1 - done_t) * V(s_{t+1}) - V(s_t)
        # A_t = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}
        deltas = rewards + self.gamma * (1.0 - dones) * next_values - values
        advantages = torch.zeros_like(rewards, device=self.device)

        gae = 0.0
        for t in reversed(range(rewards.shape[0])):
            gae = deltas[t] + self.gamma * self.gae_lambda * (1.0 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        return advantages, returns

    def _policy_loss(
        self,
        new_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        # PPO clipped surrogate objective:
        # L_clip = E[min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t)]
        # We minimize the negative objective in gradient descent.
        ratio = torch.exp(new_log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
        surrogate_1 = ratio * advantages.detach()
        surrogate_2 = clipped_ratio * advantages.detach()
        return -torch.min(surrogate_1, surrogate_2).mean()

    def _value_loss(self, new_values: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        # Value-function regression.
        return F.mse_loss(new_values, returns.detach())

    def _total_loss(
        self, policy_loss: torch.Tensor, value_loss: torch.Tensor, entropy: torch.Tensor
    ) -> torch.Tensor:
        # Minimize:
        # L = L_policy + c1 * L_value - c2 * entropy
        return policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

    def act(self, state: int, deterministic: bool = True) -> int:
        with torch.no_grad():
            logits, _ = self.net(self._state_tensor(state))
            if deterministic:
                return int(torch.argmax(logits, dim=-1).item())
            dist = Categorical(logits=logits)
            return int(dist.sample().item())

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "net": self.net.state_dict(),
            "rows": self.rows,
            "cols": self.cols,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_eps": self.clip_eps,
            "value_coef": self.value_coef,
            "entropy_coef": self.entropy_coef,
        }
        torch.save(payload, path)

    def load(self, path: str) -> None:
        payload = torch.load(path, map_location=self.device)
        self.net.load_state_dict(payload["net"])
        self.rows = int(payload.get("rows", self.rows))
        self.cols = int(payload.get("cols", self.cols))
        self.gamma = float(payload.get("gamma", self.gamma))
        self.gae_lambda = float(payload.get("gae_lambda", self.gae_lambda))
        self.clip_eps = float(payload.get("clip_eps", self.clip_eps))
        self.value_coef = float(payload.get("value_coef", self.value_coef))
        self.entropy_coef = float(payload.get("entropy_coef", self.entropy_coef))

    def get_policy(self) -> np.ndarray:
        actions = np.zeros(self.state_space, dtype=np.int64)
        with torch.no_grad():
            for state in range(self.state_space):
                logits, _ = self.net(self._state_tensor(state))
                actions[state] = int(torch.argmax(logits, dim=-1).item())
        return actions
