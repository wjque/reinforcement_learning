from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm

from common.base_agent import BaseAgent
from common.buffers import A2CBuffer
from common.networks import SharedPolicyValueNet


class ActorCriticAgent(BaseAgent):
    algorithm_name = "actor_critic"
    checkpoint_type = "pt"

    def __init__(
        self,
        obs_dim: int,
        action_space: int,
        *,
        gamma: float = 0.99,
        learning_rate: float = 3e-4,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        grad_clip_norm: float = 0.5,
        rollout_steps: int = 256,
        hidden_dims: tuple[int, ...] = (128, 128),
        device: str = "cpu",
    ) -> None:
        super().__init__(obs_dim, action_space)
        self.gamma = gamma
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.grad_clip_norm = grad_clip_norm
        self.rollout_steps = rollout_steps
        self.device = torch.device(device)

        self.net = SharedPolicyValueNet(
            input_dim=obs_dim,
            action_dim=action_space,
            hidden_dims=hidden_dims,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)

    def _obs_tensor(self, obs: np.ndarray) -> torch.Tensor:
        return torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

    def _sample_action(self, obs: np.ndarray) -> int:
        with torch.no_grad():
            logits, _value = self.net(self._obs_tensor(obs))
            dist = Categorical(logits=logits)
            return int(dist.sample().item())

    def train(
        self,
        env: Any,
        total_steps: int = 300_000,
        seed: int | None = None,
    ) -> Dict[str, Any]:
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        buffer = A2CBuffer()
        episode_returns: list[float] = []
        actor_losses: list[float] = []
        critic_losses: list[float] = []
        entropy_values: list[float] = []

        left_obs, right_obs = env.reset(seed=seed)
        current_episode_return = 0.0
        episodes_finished = 0

        for step_idx in tqdm(range(1, total_steps + 1), desc="A2C training"):
            left_action = self._sample_action(left_obs)
            right_action = self._sample_action(right_obs)

            (next_left_obs, next_right_obs), (left_reward, right_reward), done, _info = env.step(
                left_action,
                right_action,
            )

            buffer.add(left_obs, left_action, float(left_reward), bool(done), next_left_obs)
            buffer.add(right_obs, right_action, float(right_reward), bool(done), next_right_obs)

            current_episode_return += float(0.5 * (left_reward + right_reward))
            left_obs, right_obs = next_left_obs, next_right_obs

            should_update = len(buffer) >= self.rollout_steps * 2 or done or step_idx == total_steps
            if should_update:
                metrics = self._update(buffer)
                buffer.clear()
                if metrics:
                    actor_losses.append(metrics["actor_loss"])
                    critic_losses.append(metrics["critic_loss"])
                    entropy_values.append(metrics["entropy"])

            if done:
                episode_returns.append(current_episode_return)
                current_episode_return = 0.0
                episodes_finished += 1
                next_seed = None if seed is None else seed + episodes_finished
                left_obs, right_obs = env.reset(seed=next_seed)

        return {
            "episode_returns": episode_returns,
            "actor_loss": actor_losses,
            "critic_loss": critic_losses,
            "entropy": entropy_values,
            "episodes": episodes_finished,
            "total_steps": total_steps,
        }

    def _update(self, buffer: A2CBuffer) -> dict[str, float]:
        if len(buffer) == 0:
            return {}

        obs_t = torch.tensor(np.array(buffer.observations), dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(buffer.actions, dtype=torch.long, device=self.device)
        rewards_t = torch.tensor(buffer.rewards, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(buffer.dones, dtype=torch.float32, device=self.device)
        next_obs_t = torch.tensor(np.array(buffer.next_observations), dtype=torch.float32, device=self.device)

        logits, values = self.net(obs_t)
        values = values.squeeze(-1)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions_t)
        entropy = dist.entropy().mean()

        with torch.no_grad():
            _, next_values = self.net(next_obs_t)
            next_values = next_values.squeeze(-1)

        td_target = rewards_t + self.gamma * (1.0 - dones_t) * next_values
        advantage = td_target - values
        advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)

        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = F.mse_loss(values, td_target.detach())
        total_loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip_norm)
        self.optimizer.step()

        return {
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "entropy": float(entropy.item()),
        }

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
            "value_coef": self.value_coef,
            "entropy_coef": self.entropy_coef,
            "grad_clip_norm": self.grad_clip_norm,
            "rollout_steps": self.rollout_steps,
        }
        torch.save(payload, path)

    def load(self, path: str) -> None:
        payload = torch.load(path, map_location=self.device)
        self.net.load_state_dict(payload["net"])
        self.gamma = float(payload.get("gamma", self.gamma))
        self.value_coef = float(payload.get("value_coef", self.value_coef))
        self.entropy_coef = float(payload.get("entropy_coef", self.entropy_coef))
        self.grad_clip_norm = float(payload.get("grad_clip_norm", self.grad_clip_norm))
        self.rollout_steps = int(payload.get("rollout_steps", self.rollout_steps))

