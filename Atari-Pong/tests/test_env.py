from __future__ import annotations

import unittest

import numpy as np

from env.pong_env import PongEnv


class TestPongEnv(unittest.TestCase):
    def test_reset_returns_two_observations(self) -> None:
        env = PongEnv()
        left_obs, right_obs = env.reset(seed=123)
        self.assertEqual(left_obs.shape, (env.obs_dim,))
        self.assertEqual(right_obs.shape, (env.obs_dim,))
        self.assertTrue(np.all(left_obs[:2] >= 0.0))
        self.assertTrue(np.all(left_obs[:2] <= 1.0))

    def test_seed_reproducibility(self) -> None:
        env = PongEnv()
        left_1, right_1 = env.reset(seed=7)
        left_2, right_2 = env.reset(seed=7)
        np.testing.assert_allclose(left_1, left_2, atol=1e-7)
        np.testing.assert_allclose(right_1, right_2, atol=1e-7)

    def test_mirror_observation_consistency(self) -> None:
        env = PongEnv()
        left_obs, right_obs = env.reset(seed=11)
        self.assertAlmostEqual(float(left_obs[0] + right_obs[0]), 1.0, places=5)
        self.assertAlmostEqual(float(left_obs[2] + right_obs[2]), 0.0, places=5)

    def test_left_paddle_return_reward(self) -> None:
        env = PongEnv(target_score=5)
        env.reset(seed=0)
        env.ball_y = env.left_paddle_y
        env.ball_x = env.left_paddle_x + env.paddle_w / 2.0 + env.ball_r + 0.002
        env.ball_vx = -0.02
        env.ball_vy = 0.0

        (_left_obs, _right_obs), (left_reward, right_reward), done, info = env.step(PongEnv.STAY, PongEnv.STAY)
        self.assertFalse(done)
        self.assertEqual(info["event"], "left_return")
        self.assertGreater(left_reward, env.reward_step)
        self.assertAlmostEqual(right_reward, env.reward_step, places=6)

    def test_score_and_target_done(self) -> None:
        env = PongEnv(target_score=1)
        env.reset(seed=0)
        env.ball_y = env.height * 0.5
        env.ball_x = 0.001
        env.ball_vx = -0.03
        env.ball_vy = 0.0

        (_left_obs, _right_obs), (left_reward, right_reward), done, info = env.step(PongEnv.STAY, PongEnv.STAY)
        self.assertTrue(done)
        self.assertEqual(info["event"], "target_score")
        self.assertEqual(env.score_right, 1)
        self.assertLess(left_reward, 0.0)
        self.assertGreater(right_reward, 0.0)

    def test_max_steps_done(self) -> None:
        env = PongEnv(target_score=99, max_steps=1)
        env.reset(seed=0)
        (_left_obs, _right_obs), _rewards, done, info = env.step(PongEnv.STAY, PongEnv.STAY)
        self.assertTrue(done)
        self.assertEqual(info["event"], "max_steps")

    def test_copy_is_independent(self) -> None:
        env = PongEnv(target_score=7)
        env.reset(seed=2)
        env.step(PongEnv.UP, PongEnv.DOWN)
        clone = env.copy()

        self.assertEqual(clone.score_left, env.score_left)
        self.assertEqual(clone.score_right, env.score_right)
        self.assertEqual(clone.steps, env.steps)
        self.assertAlmostEqual(clone.ball_x, env.ball_x, places=7)

        clone.step(PongEnv.STAY, PongEnv.STAY)
        self.assertNotEqual(clone.steps, env.steps)


if __name__ == "__main__":
    unittest.main()

