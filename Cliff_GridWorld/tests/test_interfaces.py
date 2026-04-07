from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from common.checkpoints import SUPPORTED_ALGOS, checkpoint_filename
from common.factory import create_agent
from env.cliff_gridworld import CliffGridWorldEnv


class TestAgentInterfaces(unittest.TestCase):
    def setUp(self) -> None:
        self.env = CliffGridWorldEnv()

    def test_agents_support_shared_interface_and_checkpoint_roundtrip(self) -> None:
        has_torch = True
        try:
            import torch  # noqa: F401
        except ModuleNotFoundError:
            has_torch = False

        algos = list(SUPPORTED_ALGOS)
        if not has_torch:
            algos = [a for a in algos if a not in {"actor_critic", "ppo"}]

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for algo in algos:
                agent = create_agent(
                    algo,
                    self.env.state_space,
                    self.env.action_space,
                    rows=self.env.rows,
                    cols=self.env.cols,
                )
                self.assertTrue(hasattr(agent, "train"))
                self.assertTrue(hasattr(agent, "act"))
                self.assertTrue(hasattr(agent, "save"))
                self.assertTrue(hasattr(agent, "load"))
                self.assertTrue(hasattr(agent, "get_policy"))

                ckpt = root / checkpoint_filename(algo)
                agent.save(str(ckpt))
                self.assertTrue(ckpt.exists())

                loaded = create_agent(
                    algo,
                    self.env.state_space,
                    self.env.action_space,
                    rows=self.env.rows,
                    cols=self.env.cols,
                )
                loaded.load(str(ckpt))

                policy = loaded.get_policy()
                self.assertIsInstance(policy, np.ndarray)
                self.assertEqual(policy.shape, (self.env.state_space,))


if __name__ == "__main__":
    unittest.main()
