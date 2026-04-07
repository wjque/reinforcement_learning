from __future__ import annotations

import unittest

from env.pong_env import PongEnv
from play import human_action_from_flags, launch_human_vs_agent, normalize_human_side, run_agent_vs_agent


class DummyAgent:
    def __init__(self, action: int = PongEnv.STAY) -> None:
        self.action = action

    def act(self, _obs, deterministic: bool = True) -> int:
        return int(self.action)


class StubHumanRunner:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "episodes": float(kwargs.get("episodes", 1)),
            "left_win_rate": 0.0,
            "right_win_rate": 1.0,
            "draw_rate": 0.0,
        }


class TestPlayModes(unittest.TestCase):
    def test_run_agent_vs_agent_metrics_shape(self) -> None:
        env = PongEnv(target_score=3, max_steps=120)
        left_agent = DummyAgent(action=PongEnv.STAY)
        right_agent = DummyAgent(action=PongEnv.STAY)

        metrics = run_agent_vs_agent(
            left_agent=left_agent,
            right_agent=right_agent,
            env=env,
            episodes=3,
            deterministic=True,
            seed=9,
            render="none",
        )
        self.assertIn("left_win_rate", metrics)
        self.assertIn("right_win_rate", metrics)
        self.assertIn("draw_rate", metrics)
        self.assertIn("avg_left_score", metrics)
        self.assertIn("avg_right_score", metrics)
        self.assertIn("avg_steps", metrics)

    def test_human_action_mapping(self) -> None:
        self.assertEqual(human_action_from_flags(True, False), PongEnv.UP)
        self.assertEqual(human_action_from_flags(False, True), PongEnv.DOWN)
        self.assertEqual(human_action_from_flags(False, False), PongEnv.STAY)
        self.assertEqual(human_action_from_flags(True, True), PongEnv.STAY)

    def test_human_mode_launch_with_stub_runner(self) -> None:
        env = PongEnv()
        agent = DummyAgent()
        runner = StubHumanRunner()

        metrics = launch_human_vs_agent(
            agent=agent,
            env=env,
            human_side="right",
            deterministic=False,
            fps=45,
            episodes=2,
            runner=runner,
        )

        self.assertEqual(metrics["episodes"], 2.0)
        self.assertEqual(len(runner.calls), 1)
        call = runner.calls[0]
        self.assertEqual(call["human_side"], "right")
        self.assertEqual(call["fps"], 45)
        self.assertEqual(call["deterministic"], False)

    def test_normalize_human_side(self) -> None:
        self.assertEqual(normalize_human_side("left"), "left")
        self.assertEqual(normalize_human_side("RIGHT"), "right")
        with self.assertRaises(ValueError):
            normalize_human_side("center")


if __name__ == "__main__":
    unittest.main()

