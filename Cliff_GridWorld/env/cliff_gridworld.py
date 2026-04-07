from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, Tuple

import numpy as np


@dataclass(frozen=True)
class GridPos:
    row: int
    col: int


class CliffGridWorldEnv:
    """Deterministic 4x12 Cliff GridWorld environment."""

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    def __init__(
        self,
        rows: int = 4,
        cols: int = 12,
        start: Tuple[int, int] = (3, 0),
        goal: Tuple[int, int] = (3, 11),
    ) -> None:
        self.rows = rows
        self.cols = cols
        self.start = GridPos(*start)
        self.goal = GridPos(*goal)

        self.cliff_cells = {(3, c) for c in range(1, 11)}
        self.action_space = 4
        self.state_space = self.rows * self.cols
        self.max_steps = self.rows * self.cols * 4

        self._action_deltas = {
            self.UP: (-1, 0),
            self.DOWN: (1, 0),
            self.LEFT: (0, -1),
            self.RIGHT: (0, 1),
        }

        self._rng = np.random.default_rng()
        self.agent_pos = self.start
        self.steps = 0
        self.done = False

    def reset(self, seed: int | None = None) -> int:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.agent_pos = self.start
        self.steps = 0
        self.done = False
        return self.pos_to_state(self.agent_pos.row, self.agent_pos.col)

    def step(self, action: int) -> Tuple[int, float, bool, Dict[str, object]]:
        if action not in self._action_deltas:
            raise ValueError(f"Invalid action {action}. Must be in [0, 1, 2, 3].")
        if self.done:
            raise RuntimeError("Episode is done. Call reset() before calling step().")

        row, col = self.agent_pos.row, self.agent_pos.col
        delta_row, delta_col = self._action_deltas[action]
        next_row = int(np.clip(row + delta_row, 0, self.rows - 1))
        next_col = int(np.clip(col + delta_col, 0, self.cols - 1))
        next_pos = GridPos(next_row, next_col)

        reward = -1.0
        done = False
        event = "step"

        if (next_pos.row, next_pos.col) in self.cliff_cells:
            reward = -100.0
            done = True
            event = "cliff"
        elif (next_pos.row, next_pos.col) == (self.goal.row, self.goal.col):
            reward = 100.0
            done = True
            event = "goal"

        self.agent_pos = next_pos
        self.steps += 1
        if not done and self.steps >= self.max_steps:
            done = True
            event = "timeout"
        self.done = done

        next_state = self.pos_to_state(next_pos.row, next_pos.col)
        info: Dict[str, object] = {
            "event": event,
            "position": (next_pos.row, next_pos.col),
            "steps": self.steps,
        }
        return next_state, reward, done, info

    def transition(self, state: int, action: int) -> Tuple[int, float, bool, Dict[str, object]]:
        """Model-based transition for DP algorithms."""
        if self.is_terminal(state):
            return state, 0.0, True, {"event": "terminal"}

        row, col = self.state_to_pos(state)
        delta_row, delta_col = self._action_deltas[action]
        next_row = int(np.clip(row + delta_row, 0, self.rows - 1))
        next_col = int(np.clip(col + delta_col, 0, self.cols - 1))
        next_state = self.pos_to_state(next_row, next_col)

        reward = -1.0
        done = False
        event = "step"
        if (next_row, next_col) in self.cliff_cells:
            reward = -100.0
            done = True
            event = "cliff"
        elif (next_row, next_col) == (self.goal.row, self.goal.col):
            reward = 100.0
            done = True
            event = "goal"

        return next_state, reward, done, {"event": event, "position": (next_row, next_col)}

    def is_terminal(self, state: int) -> bool:
        row, col = self.state_to_pos(state)
        return (row, col) in self.cliff_cells or (row, col) == (self.goal.row, self.goal.col)

    def pos_to_state(self, row: int, col: int) -> int:
        return row * self.cols + col

    def state_to_pos(self, state: int) -> Tuple[int, int]:
        if state < 0 or state >= self.state_space:
            raise ValueError(f"State {state} out of range [0, {self.state_space - 1}].")
        row = state // self.cols
        col = state % self.cols
        return row, col

    def state_features(self, state: int) -> np.ndarray:
        """Return a compact 2D feature vector for neural agents."""
        row, col = self.state_to_pos(state)
        row_den = max(1, self.rows - 1)
        col_den = max(1, self.cols - 1)
        return np.array([row / row_den, col / col_den], dtype=np.float32)

    def iter_states(self) -> Iterator[int]:
        for s in range(self.state_space):
            yield s

