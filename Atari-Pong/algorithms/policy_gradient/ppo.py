from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm

from common.base_agent import BaseAgent
from common.buffers import PPOBuffer
from common.networks import SharedPolicyValueNet


class PPOAgent(BaseAgent):
    algorithm_name = "ppo"
    checkpoint_type = "pt"

    def __init__(
        self,
        obs_dim: int,
        action_space: int,
        *,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        learning_rate: float = 3e-4,
        grad_clip_norm: float = 0.5,
        rollout_steps: int = 512,
        ppo_epochs: int = 4,
        minibatch_size: int = 256,
        hidden_dims: tuple[int, ...] = (128, 128),
        device: str = "cpu",
    ) -> None:
        super().__init__(obs_dim, action_space)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.grad_clip_norm = grad_clip_norm
        self.rollout_steps = rollout_steps
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size
        self.device = torch.device(device)

        self.net = SharedPolicyValueNet(
            input_dim=obs_dim,
            action_dim=action_space,
            hidden_dims=hidden_dims,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)

    def _obs_tensor(self, obs: np.ndarray) -> torch.Tensor:
        return torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

    def _sample_action(self, obs: np.ndarray) -> tuple[int, float, float]:
        with torch.no_grad():
            logits, value = self.net(self._obs_tensor(obs))
            dist = Categorical(logits=logits)
            action_t = dist.sample()
            log_prob = dist.log_prob(action_t)
            return int(action_t.item()), float(log_prob.item()), float(value.squeeze(-1).item())

    def train(
        self,
        env: Any,
        total_steps: int = 300_000,
        seed: int | None = None,
    ) -> Dict[str, Any]:
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        buffer = PPOBuffer()
        episode_returns: list[float] = []
        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropy_values: list[float] = []

        left_obs, right_obs = env.reset(seed=seed)
        current_episode_return = 0.0
        episodes_finished = 0

        for step_idx in tqdm(range(1, total_steps + 1), desc="PPO training"):
            left_action, left_log_prob, left_value = self._sample_action(left_obs)
            right_action, right_log_prob, right_value = self._sample_action(right_obs)

            (next_left_obs, next_right_obs), (left_reward, right_reward), done, _info = env.step(
                left_action,
                right_action,
            )

            buffer.add(
                observation=left_obs,
                action=left_action,
                reward=float(left_reward),
                done=bool(done),
                log_prob=left_log_prob,
                value=left_value,
                next_observation=next_left_obs,
            )
            buffer.add(
                observation=right_obs,
                action=right_action,
                reward=float(right_reward),
                done=bool(done),
                log_prob=right_log_prob,
                value=right_value,
                next_observation=next_right_obs,
            )

            current_episode_return += float(0.5 * (left_reward + right_reward))
            left_obs, right_obs = next_left_obs, next_right_obs

            should_update = len(buffer) >= self.rollout_steps * 2 or done or step_idx == total_steps
            if should_update:
                metrics = self._update(buffer)
                buffer.clear()
                if metrics:
                    policy_losses.append(metrics["policy_loss"])
                    value_losses.append(metrics["value_loss"])
                    entropy_values.append(metrics["entropy"])

            if done:
                episode_returns.append(current_episode_return)
                current_episode_return = 0.0
                episodes_finished += 1
                next_seed = None if seed is None else seed + episodes_finished
                left_obs, right_obs = env.reset(seed=next_seed)

        return {
            "episode_returns": episode_returns,
            "policy_loss": policy_losses,
            "value_loss": value_losses,
            "entropy": entropy_values,
            "episodes": episodes_finished,
            "total_steps": total_steps,
        }

    def _update(self, buffer: PPOBuffer) -> dict[str, float]:
        if len(buffer) == 0:
            return {}

        obs_t = torch.tensor(np.array(buffer.observations), dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(buffer.actions, dtype=torch.long, device=self.device)
        rewards_t = torch.tensor(buffer.rewards, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(buffer.dones, dtype=torch.float32, device=self.device)
        old_log_probs_t = torch.tensor(buffer.log_probs, dtype=torch.float32, device=self.device)
        values_t = torch.tensor(buffer.values, dtype=torch.float32, device=self.device)
        next_obs_t = torch.tensor(np.array(buffer.next_observations), dtype=torch.float32, device=self.device)

        with torch.no_grad():
            _, next_values_t = self.net(next_obs_t)
            next_values_t = next_values_t.squeeze(-1)
            advantages_t, returns_t = self._compute_gae(
                rewards=rewards_t,
                dones=dones_t,
                values=values_t,
                next_values=next_values_t,
            )

        batch_size = obs_t.shape[0]
        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropy_values: list[float] = []

        for _ in range(self.ppo_epochs):
            indices = np.random.permutation(batch_size)
            for start in range(0, batch_size, self.minibatch_size):
                mb_idx = indices[start : start + self.minibatch_size]
                mb_idx_t = torch.tensor(mb_idx, dtype=torch.long, device=self.device)

                mb_obs = obs_t[mb_idx_t]
                mb_actions = actions_t[mb_idx_t]
                mb_old_log_probs = old_log_probs_t[mb_idx_t]
                mb_advantages = advantages_t[mb_idx_t]
                mb_returns = returns_t[mb_idx_t]

                logits, new_values = self.net(mb_obs)
                new_values = new_values.squeeze(-1)
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
                surrogate_1 = ratio * mb_advantages.detach()
                surrogate_2 = clipped_ratio * mb_advantages.detach()
                policy_loss = -torch.min(surrogate_1, surrogate_2).mean()
                value_loss = F.mse_loss(new_values, mb_returns.detach())
                total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip_norm)
                self.optimizer.step()

                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropy_values.append(float(entropy.item()))

        return {
            "policy_loss": float(np.mean(policy_losses) if policy_losses else 0.0),
            "value_loss": float(np.mean(value_losses) if value_losses else 0.0),
            "entropy": float(np.mean(entropy_values) if entropy_values else 0.0),
        }

    def _compute_gae(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        deltas = rewards + self.gamma * (1.0 - dones) * next_values - values
        advantages = torch.zeros_like(deltas, device=self.device)

        gae = torch.tensor(0.0, device=self.device)
        for t in reversed(range(deltas.shape[0])):
            gae = deltas[t] + self.gamma * self.gae_lambda * (1.0 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        return advantages, returns

    def act(self, obs: np.ndarray, deterministic: bool = True) -> int:
        with torch.no_grad():
            logits, _value = self.net(self._obs_tensor(obs))
            if deterministic:
                return int(torch.argmax(logits, dim=-1).item())
            dist = Categorical(logits=logits)
            return int(dist.sample().item())

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "net": self.net.state_dict(),
            "obs_dim": self.obs_dim,
            "action_space": self.action_space,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_eps": self.clip_eps,
            "value_coef": self.value_coef,
            "entropy_coef": self.entropy_coef,
            "grad_clip_norm": self.grad_clip_norm,
            "rollout_steps": self.rollout_steps,
            "ppo_epochs": self.ppo_epochs,
            "minibatch_size": self.minibatch_size,
        }
        torch.save(payload, path)

    def load(self, path: str) -> None:
        payload = torch.load(path, map_location=self.device)
        self.net.load_state_dict(payload["net"])
        self.gamma = float(payload.get("gamma", self.gamma))
        self.gae_lambda = float(payload.get("gae_lambda", self.gae_lambda))
        self.clip_eps = float(payload.get("clip_eps", self.clip_eps))
        self.value_coef = float(payload.get("value_coef", self.value_coef))
        self.entropy_coef = float(payload.get("entropy_coef", self.entropy_coef))
        self.grad_clip_norm = float(payload.get("grad_clip_norm", self.grad_clip_norm))
        self.rollout_steps = int(payload.get("rollout_steps", self.rollout_steps))
        self.ppo_epochs = int(payload.get("ppo_epochs", self.ppo_epochs))
        self.minibatch_size = int(payload.get("minibatch_size", self.minibatch_size))

