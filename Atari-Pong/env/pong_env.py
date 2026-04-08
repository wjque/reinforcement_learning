from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np


@dataclass
class PongSnapshot:
    width: float
    height: float
    left_paddle_x: float
    right_paddle_x: float
    left_paddle_y: float
    right_paddle_y: float
    paddle_h: float
    paddle_w: float
    ball_x: float
    ball_y: float
    ball_r: float
    score_left: int
    score_right: int
    steps: int
    target_score: int
    event: str


class PongEnv:
    """Simplified two-player Pong environment for shared-policy self-play."""

    UP = 0
    STAY = 1
    DOWN = 2

    def __init__(
        self,
        *,
        width: float = 1.0,
        height: float = 1.0,
        target_score: int = 11,
        max_steps: int = 2000,
        paddle_h: float = 0.22,
        paddle_w: float = 0.02,
        paddle_speed: float = 0.035,
        ball_r: float = 0.015,
        ball_speed: float = 0.018,
        max_ball_speed: float = 0.04,
        reward_score: float = 1.0,
        reward_concede: float = -1.0,
        reward_return: float = 0.02,
        reward_step: float = -0.001,
    ) -> None:
        self.width = float(width)
        self.height = float(height)
        self.target_score = int(target_score)
        self.max_steps = int(max_steps)

        self.paddle_h = float(paddle_h)
        self.paddle_w = float(paddle_w)
        self.paddle_speed = float(paddle_speed)
        self.ball_r = float(ball_r)
        self.ball_speed = float(ball_speed)
        self.max_ball_speed = float(max_ball_speed)

        self.reward_score = float(reward_score)
        self.reward_concede = float(reward_concede)
        self.reward_return = float(reward_return)
        self.reward_step = float(reward_step)

        self.left_paddle_x = 0.06
        self.right_paddle_x = self.width - self.left_paddle_x

        self.action_space = 3
        self.obs_dim = 10

        self._rng = np.random.default_rng()
        self.render_hook: Callable[[PongSnapshot], None] | None = None
        self.last_event = "reset"

        self.score_left = 0
        self.score_right = 0
        self.steps = 0
        self.done = False

        self.left_paddle_y = self.height / 2.0
        self.right_paddle_y = self.height / 2.0
        self.ball_x = self.width / 2.0
        self.ball_y = self.height / 2.0
        self.ball_vx = 0.0
        self.ball_vy = 0.0
        self._reset_ball(direction=1)

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.score_left = 0
        self.score_right = 0
        self.steps = 0
        self.done = False
        self.left_paddle_y = self.height / 2.0
        self.right_paddle_y = self.height / 2.0
        self.last_event = "reset"
        self._reset_ball(direction=int(self._rng.choice([-1, 1])))
        self._emit_render()
        return self._get_observations()

    def step(
        self,
        left_action: int,
        right_action: int,
    ) -> tuple[tuple[np.ndarray, np.ndarray], tuple[float, float], bool, Dict[str, object]]:
        if self.done:
            raise RuntimeError("Episode already done. Call reset() before step().")
        if left_action not in {self.UP, self.STAY, self.DOWN}:
            raise ValueError(f"Invalid left_action {left_action}. Must be in [0, 1, 2].")
        if right_action not in {self.UP, self.STAY, self.DOWN}:
            raise ValueError(f"Invalid right_action {right_action}. Must be in [0, 1, 2].")

        self.left_paddle_y = self._apply_paddle_action(self.left_paddle_y, left_action)
        self.right_paddle_y = self._apply_paddle_action(self.right_paddle_y, right_action)

        reward_left = self.reward_step
        reward_right = self.reward_step
        event = "running"
        point_scored = False

        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        if self.ball_y - self.ball_r <= 0.0:
            self.ball_y = self.ball_r
            self.ball_vy = abs(self.ball_vy)
        elif self.ball_y + self.ball_r >= self.height:
            self.ball_y = self.height - self.ball_r
            self.ball_vy = -abs(self.ball_vy)

        if self._check_left_paddle_collision():
            reward_left += self.reward_return
            self.ball_x = self.left_paddle_x + self.paddle_w / 2.0 + self.ball_r
            self.ball_vx = abs(self.ball_vx) * 1.02
            self.ball_vx = min(self.ball_vx, self.max_ball_speed)
            self._inject_spin(self.left_paddle_y)
            event = "left_return"
        elif self._check_right_paddle_collision():
            reward_right += self.reward_return
            self.ball_x = self.right_paddle_x - self.paddle_w / 2.0 - self.ball_r
            self.ball_vx = -abs(self.ball_vx) * 1.02
            self.ball_vx = max(self.ball_vx, -self.max_ball_speed)
            self._inject_spin(self.right_paddle_y)
            event = "right_return"

        if self.ball_x < 0.0:
            self.score_right += 1
            reward_left += self.reward_concede
            reward_right += self.reward_score
            point_scored = True
            event = "right_score"
            self._reset_ball(direction=-1)
        elif self.ball_x > self.width:
            self.score_left += 1
            reward_left += self.reward_score
            reward_right += self.reward_concede
            point_scored = True
            event = "left_score"
            self._reset_ball(direction=1)

        self.steps += 1
        done = False
        if self.score_left >= self.target_score or self.score_right >= self.target_score:
            done = True
            event = "target_score"
        elif self.steps >= self.max_steps:
            done = True
            event = "max_steps"

        self.done = done
        self.last_event = event

        observations = self._get_observations()
        info: Dict[str, object] = {
            "event": event,
            "point_scored": point_scored,
            "score_left": self.score_left,
            "score_right": self.score_right,
            "steps": self.steps,
        }
        self._emit_render()
        return observations, (reward_left, reward_right), done, info

    def snapshot(self) -> PongSnapshot:
        return PongSnapshot(
            width=self.width,
            height=self.height,
            left_paddle_x=self.left_paddle_x,
            right_paddle_x=self.right_paddle_x,
            left_paddle_y=self.left_paddle_y,
            right_paddle_y=self.right_paddle_y,
            paddle_h=self.paddle_h,
            paddle_w=self.paddle_w,
            ball_x=self.ball_x,
            ball_y=self.ball_y,
            ball_r=self.ball_r,
            score_left=self.score_left,
            score_right=self.score_right,
            steps=self.steps,
            target_score=self.target_score,
            event=self.last_event,
        )

    def copy(self) -> PongEnv:
        cloned = PongEnv(
            width=self.width,
            height=self.height,
            target_score=self.target_score,
            max_steps=self.max_steps,
            paddle_h=self.paddle_h,
            paddle_w=self.paddle_w,
            paddle_speed=self.paddle_speed,
            ball_r=self.ball_r,
            ball_speed=self.ball_speed,
            max_ball_speed=self.max_ball_speed,
            reward_score=self.reward_score,
            reward_concede=self.reward_concede,
            reward_return=self.reward_return,
            reward_step=self.reward_step,
        )
        cloned.score_left = self.score_left
        cloned.score_right = self.score_right
        cloned.steps = self.steps
        cloned.done = self.done
        cloned.left_paddle_y = self.left_paddle_y
        cloned.right_paddle_y = self.right_paddle_y
        cloned.ball_x = self.ball_x
        cloned.ball_y = self.ball_y
        cloned.ball_vx = self.ball_vx
        cloned.ball_vy = self.ball_vy
        cloned.last_event = self.last_event
        cloned.render_hook = self.render_hook
        cloned._rng = np.random.default_rng()
        cloned._rng.bit_generator.state = self._rng.bit_generator.state
        return cloned

    def _apply_paddle_action(self, paddle_y: float, action: int) -> float:
        if action == self.UP:
            paddle_y -= self.paddle_speed
        elif action == self.DOWN:
            paddle_y += self.paddle_speed
        min_y = self.paddle_h / 2.0
        max_y = self.height - self.paddle_h / 2.0
        return float(np.clip(paddle_y, min_y, max_y))

    def _check_left_paddle_collision(self) -> bool:
        return (
            self.ball_vx < 0.0
            and self.ball_x - self.ball_r <= self.left_paddle_x + self.paddle_w / 2.0
            and self.ball_x > self.left_paddle_x - self.paddle_w / 2.0
            and abs(self.ball_y - self.left_paddle_y) <= self.paddle_h / 2.0 + self.ball_r
        )

    def _check_right_paddle_collision(self) -> bool:
        return (
            self.ball_vx > 0.0
            and self.ball_x + self.ball_r >= self.right_paddle_x - self.paddle_w / 2.0
            and self.ball_x < self.right_paddle_x + self.paddle_w / 2.0
            and abs(self.ball_y - self.right_paddle_y) <= self.paddle_h / 2.0 + self.ball_r
        )

    def _inject_spin(self, paddle_y: float) -> None:
        offset = (self.ball_y - paddle_y) / max(self.paddle_h / 2.0, 1e-6)
        self.ball_vy += 0.012 * float(np.clip(offset, -1.0, 1.0))
        self.ball_vy = float(np.clip(self.ball_vy, -self.max_ball_speed, self.max_ball_speed))

    def _reset_ball(self, direction: int) -> None:
        self.ball_x = self.width / 2.0
        self.ball_y = float(self._rng.uniform(0.25 * self.height, 0.75 * self.height))
        vy_scale = float(self._rng.uniform(-0.65, 0.65))
        self.ball_vx = float(direction) * self.ball_speed
        self.ball_vy = vy_scale * self.ball_speed

    def _observe_side(self, side: str) -> np.ndarray:
        if side not in {"left", "right"}:
            raise ValueError(f"Unknown side '{side}'.")

        if side == "left":
            ball_x = self.ball_x / self.width
            ball_vx = self.ball_vx
            self_paddle = self.left_paddle_y / self.height
            opp_paddle = self.right_paddle_y / self.height
            score_self = self.score_left
            score_opp = self.score_right
            self_x = self.left_paddle_x / self.width
        else:
            ball_x = (self.width - self.ball_x) / self.width
            ball_vx = -self.ball_vx
            self_paddle = self.right_paddle_y / self.height
            opp_paddle = self.left_paddle_y / self.height
            score_self = self.score_right
            score_opp = self.score_left
            self_x = (self.width - self.right_paddle_x) / self.width

        ball_y = self.ball_y / self.height
        ball_vx_norm = float(np.clip(ball_vx / self.max_ball_speed, -1.0, 1.0))
        ball_vy_norm = float(np.clip(self.ball_vy / self.max_ball_speed, -1.0, 1.0))
        rel_x = float(np.clip(ball_x - self_x, -1.0, 1.0))
        rel_self_y = float(np.clip(ball_y - self_paddle, -1.0, 1.0))
        rel_opp_y = float(np.clip(ball_y - opp_paddle, -1.0, 1.0))
        score_diff = float(np.clip((score_self - score_opp) / max(self.target_score, 1), -1.0, 1.0))

        obs = np.array(
            [
                float(np.clip(ball_x, 0.0, 1.0)),
                float(np.clip(ball_y, 0.0, 1.0)),
                ball_vx_norm,
                ball_vy_norm,
                float(np.clip(self_paddle, 0.0, 1.0)),
                float(np.clip(opp_paddle, 0.0, 1.0)),
                rel_x,
                rel_self_y,
                rel_opp_y,
                score_diff,
            ],
            dtype=np.float32,
        )
        return obs

    def _get_observations(self) -> tuple[np.ndarray, np.ndarray]:
        return self._observe_side("left"), self._observe_side("right")

    def _emit_render(self) -> None:
        if self.render_hook is not None:
            self.render_hook(self.snapshot())
