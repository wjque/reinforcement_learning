from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from common.checkpoints import SUPPORTED_ALGOS, checkpoint_filename
from common.factory import create_agent
from env.pong_env import PongEnv

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ModuleNotFoundError:
    HAS_TORCH = False


@unittest.skipUnless(HAS_TORCH, "PyTorch is required for policy-gradient agent tests.")
class TestPolicyGradientAgents(unittest.TestCase):
    def setUp(self) -> None:
        self.env = PongEnv(target_score=3, max_steps=240)

    def test_act_output_range(self) -> None:
        for algo in SUPPORTED_ALGOS:
            agent = create_agent(algo, self.env.obs_dim, self.env.action_space, device="cpu")
            obs_left, _obs_right = self.env.reset(seed=3)
            action = agent.act(obs_left, deterministic=True)
            self.assertIsInstance(action, int)
            self.assertTrue(0 <= action < self.env.action_space)

    def test_train_and_checkpoint_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for algo in SUPPORTED_ALGOS:
                agent = create_agent(algo, self.env.obs_dim, self.env.action_space, device="cpu")
                stats = agent.train(self.env, total_steps=256, seed=17)
                self.assertIn("episode_returns", stats)
                self.assertGreaterEqual(int(stats.get("total_steps", 0)), 256)

                ckpt = root / checkpoint_filename(algo)
                agent.save(str(ckpt))
                self.assertTrue(ckpt.exists())

                loaded = create_agent(algo, self.env.obs_dim, self.env.action_space, device="cpu")
                loaded.load(str(ckpt))
                obs_left, _obs_right = self.env.reset(seed=19)
                action = loaded.act(obs_left, deterministic=False)
                self.assertTrue(0 <= action < self.env.action_space)

    def test_network_forward_shape_via_action(self) -> None:
        agent = create_agent("ppo", self.env.obs_dim, self.env.action_space, device="cpu")
        for _ in range(5):
            obs = np.random.uniform(low=-1.0, high=1.0, size=(self.env.obs_dim,)).astype(np.float32)
            action = agent.act(obs, deterministic=False)
            self.assertTrue(0 <= action < self.env.action_space)


if __name__ == "__main__":
    unittest.main()

