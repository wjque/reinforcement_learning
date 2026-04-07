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


@unittest.skipUnless(HAS_TORCH, "PyTorch is required for interface tests.")
class TestInterfaces(unittest.TestCase):
    def setUp(self) -> None:
        self.env = PongEnv()

    def test_agents_match_common_interface(self) -> None:
        for algo in SUPPORTED_ALGOS:
            agent = create_agent(
                algo,
                self.env.obs_dim,
                self.env.action_space,
                device="cpu",
            )
            self.assertTrue(hasattr(agent, "train"))
            self.assertTrue(hasattr(agent, "act"))
            self.assertTrue(hasattr(agent, "save"))
            self.assertTrue(hasattr(agent, "load"))

    def test_checkpoint_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for algo in SUPPORTED_ALGOS:
                agent = create_agent(algo, self.env.obs_dim, self.env.action_space, device="cpu")
                ckpt = root / checkpoint_filename(algo)
                agent.save(str(ckpt))
                self.assertTrue(ckpt.exists())

                loaded = create_agent(algo, self.env.obs_dim, self.env.action_space, device="cpu")
                loaded.load(str(ckpt))
                obs_left, _obs_right = self.env.reset(seed=5)
                action = loaded.act(obs_left, deterministic=True)
                self.assertIsInstance(action, int)
                self.assertTrue(0 <= action < self.env.action_space)

    def test_act_accepts_numpy_vector(self) -> None:
        agent = create_agent("actor_critic", self.env.obs_dim, self.env.action_space, device="cpu")
        obs = np.zeros((self.env.obs_dim,), dtype=np.float32)
        action = agent.act(obs, deterministic=True)
        self.assertTrue(0 <= action < self.env.action_space)


if __name__ == "__main__":
    unittest.main()

