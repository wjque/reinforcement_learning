from __future__ import annotations

import unittest

from env.cliff_gridworld import CliffGridWorldEnv


class TestCliffGridWorldEnv(unittest.TestCase):
    def setUp(self) -> None:
        self.env = CliffGridWorldEnv()

    def test_boundary_movement(self) -> None:
        self.env.reset(seed=1)
        next_state, reward, done, info = self.env.step(self.env.DOWN)
        self.assertEqual(next_state, self.env.pos_to_state(3, 0))
        self.assertEqual(reward, -1.0)
        self.assertFalse(done)
        self.assertEqual(info["event"], "step")

        self.env.reset(seed=1)
        next_state, reward, done, info = self.env.step(self.env.LEFT)
        self.assertEqual(next_state, self.env.pos_to_state(3, 0))
        self.assertEqual(reward, -1.0)
        self.assertFalse(done)
        self.assertEqual(info["event"], "step")

    def test_cliff_penalty_and_done(self) -> None:
        self.env.reset(seed=1)
        next_state, reward, done, info = self.env.step(self.env.RIGHT)
        self.assertEqual(next_state, self.env.pos_to_state(3, 1))
        self.assertEqual(reward, -100.0)
        self.assertTrue(done)
        self.assertEqual(info["event"], "cliff")

    def test_goal_reward_and_done(self) -> None:
        self.env.reset(seed=1)
        # Move around cliff: up -> right x11 -> down.
        self.env.step(self.env.UP)
        for _ in range(11):
            self.env.step(self.env.RIGHT)
        next_state, reward, done, info = self.env.step(self.env.DOWN)

        self.assertEqual(next_state, self.env.pos_to_state(3, 11))
        self.assertEqual(reward, 100.0)
        self.assertTrue(done)
        self.assertEqual(info["event"], "goal")

    def test_state_encode_decode_roundtrip(self) -> None:
        for state in range(self.env.state_space):
            row, col = self.env.state_to_pos(state)
            self.assertEqual(self.env.pos_to_state(row, col), state)


if __name__ == "__main__":
    unittest.main()

