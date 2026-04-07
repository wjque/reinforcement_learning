from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from common.checkpoints import SUPPORTED_ALGOS, checkpoint_filename
from common.factory import create_agent
from env.cliff_gridworld import CliffGridWorldEnv
from play import run_play


class TestPlayIntegration(unittest.TestCase):
    def test_run_play_with_existing_checkpoints(self) -> None:
        env = CliffGridWorldEnv()
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
            model_dir = root / "checkpoints"
            out_dir = root / "outputs"
            model_dir.mkdir(parents=True, exist_ok=True)

            # Save untrained checkpoints for each algorithm to validate load/eval wiring.
            for algo in algos:
                agent = create_agent(
                    algo,
                    env.state_space,
                    env.action_space,
                    rows=env.rows,
                    cols=env.cols,
                )
                agent.save(str(model_dir / checkpoint_filename(algo)))

            results = run_play(
                algos=algos,
                model_dir=str(model_dir),
                output_dir=str(out_dir),
                episodes=2,
                deterministic=True,
            )

            self.assertEqual(set(results.keys()), set(algos))
            for algo in algos:
                self.assertIn("avg_return", results[algo])
                self.assertIn("success_rate", results[algo])
                self.assertIn("avg_steps", results[algo])
                self.assertIn("cliff_rate", results[algo])
                self.assertTrue((out_dir / f"policy_{algo}.png").exists())
                self.assertTrue((out_dir / f"trajectory_{algo}.png").exists())

    def test_run_play_raises_on_missing_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "checkpoints"
            out_dir = root / "outputs"
            model_dir.mkdir(parents=True, exist_ok=True)

            with self.assertRaises(FileNotFoundError):
                run_play(
                    algos=["sarsa"],
                    model_dir=str(model_dir),
                    output_dir=str(out_dir),
                    episodes=1,
                    deterministic=True,
                )


if __name__ == "__main__":
    unittest.main()
