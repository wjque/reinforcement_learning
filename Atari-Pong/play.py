from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

from common.checkpoints import SUPPORTED_ALGOS, checkpoint_filename
from common.factory import create_agent
from env.pong_env import PongEnv


def normalize_human_side(side: str) -> str:
    side = side.lower().strip()
    if side not in {"left", "right"}:
        raise ValueError("human_side must be 'left' or 'right'.")
    return side


def human_action_from_flags(move_up: bool, move_down: bool) -> int:
    if move_up and not move_down:
        return PongEnv.UP
    if move_down and not move_up:
        return PongEnv.DOWN
    return PongEnv.STAY


def run_agent_vs_agent(
    left_agent,
    right_agent,
    env: PongEnv,
    *,
    episodes: int = 20,
    deterministic: bool = True,
    seed: int = 123,
    render: str = "none",
    fps: int = 60,
) -> dict[str, float]:
    left_wins = 0
    right_wins = 0
    draws = 0
    left_scores: list[int] = []
    right_scores: list[int] = []
    step_counts: list[int] = []

    live_renderer = None
    if render == "pyqt":
        from env.pyqt_renderer import PongLiveRenderer

        live_renderer = PongLiveRenderer(title="Agent vs Agent", fps=fps)
        env.render_hook = live_renderer.update

    for episode in range(episodes):
        left_obs, right_obs = env.reset(seed=seed + episode)
        done = False
        while not done:
            left_action = left_agent.act(left_obs, deterministic=deterministic)
            right_action = right_agent.act(right_obs, deterministic=deterministic)
            (left_obs, right_obs), _rewards, done, _info = env.step(left_action, right_action)

        left_score = int(env.score_left)
        right_score = int(env.score_right)
        left_scores.append(left_score)
        right_scores.append(right_score)
        step_counts.append(int(env.steps))

        if left_score > right_score:
            left_wins += 1
        elif right_score > left_score:
            right_wins += 1
        else:
            draws += 1

    if live_renderer is not None:
        live_renderer.close()
        env.render_hook = None

    eps = max(1, int(episodes))
    return {
        "left_win_rate": float(left_wins / eps),
        "right_win_rate": float(right_wins / eps),
        "draw_rate": float(draws / eps),
        "avg_left_score": float(sum(left_scores) / eps),
        "avg_right_score": float(sum(right_scores) / eps),
        "avg_steps": float(sum(step_counts) / eps),
    }


def launch_human_vs_agent(
    agent,
    env: PongEnv,
    *,
    human_side: str = "left",
    deterministic: bool = True,
    fps: int = 60,
    episodes: int = 1,
    runner: Callable[..., dict[str, float]] | None = None,
) -> dict[str, float]:
    side = normalize_human_side(human_side)
    if runner is None:
        from env.pyqt_renderer import run_human_vs_agent as runner

    return runner(
        env=env,
        agent=agent,
        human_side=side,
        deterministic=deterministic,
        fps=fps,
        episodes=episodes,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play/evaluate Atari-Pong agents.")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=("agent_vs_agent", "human_vs_agent"),
        help="Play mode.",
    )
    parser.add_argument("--model-dir", type=str, default="checkpoints", help="Checkpoint directory.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device.")
    parser.add_argument("--episodes", type=int, default=20, help="Number of episodes.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for evaluation.")
    parser.add_argument(
        "--deterministic",
        dest="deterministic",
        action="store_true",
        default=True,
        help="Use deterministic policy action selection (default).",
    )
    parser.add_argument(
        "--stochastic",
        dest="deterministic",
        action="store_false",
        help="Use stochastic policy action selection.",
    )
    parser.add_argument("--fps", type=int, default=60, help="Render/update FPS.")
    parser.add_argument("--render", type=str, choices=("none", "pyqt"), default="none")

    parser.add_argument("--left-algo", type=str, choices=SUPPORTED_ALGOS, default="ppo")
    parser.add_argument("--right-algo", type=str, choices=SUPPORTED_ALGOS, default="actor_critic")
    parser.add_argument("--left-checkpoint", type=str, default="", help="Override left checkpoint path.")
    parser.add_argument("--right-checkpoint", type=str, default="", help="Override right checkpoint path.")

    parser.add_argument("--agent-algo", type=str, choices=SUPPORTED_ALGOS, default="ppo")
    parser.add_argument("--agent-checkpoint", type=str, default="", help="Override agent checkpoint path.")
    parser.add_argument("--human-side", type=str, choices=("left", "right"), default="left")
    return parser.parse_args()


def _resolve_checkpoint(model_dir: str, algo: str, override_path: str = "") -> Path:
    if override_path:
        return Path(override_path)
    return Path(model_dir) / checkpoint_filename(algo)


def main() -> None:
    args = parse_args()
    env = PongEnv()

    if args.mode == "agent_vs_agent":
        left_agent = create_agent(args.left_algo, env.obs_dim, env.action_space, device=args.device)
        right_agent = create_agent(args.right_algo, env.obs_dim, env.action_space, device=args.device)

        left_ckpt = _resolve_checkpoint(args.model_dir, args.left_algo, args.left_checkpoint)
        right_ckpt = _resolve_checkpoint(args.model_dir, args.right_algo, args.right_checkpoint)
        if not left_ckpt.exists():
            raise FileNotFoundError(f"Left checkpoint not found: {left_ckpt}")
        if not right_ckpt.exists():
            raise FileNotFoundError(f"Right checkpoint not found: {right_ckpt}")
        left_agent.load(str(left_ckpt))
        right_agent.load(str(right_ckpt))

        metrics = run_agent_vs_agent(
            left_agent=left_agent,
            right_agent=right_agent,
            env=env,
            episodes=args.episodes,
            deterministic=args.deterministic,
            seed=args.seed,
            render=args.render,
            fps=args.fps,
        )
        print("[play] mode=agent_vs_agent")
        print(f"[play] left_algo={args.left_algo}, right_algo={args.right_algo}")
        print(
            "[play] "
            f"left_win_rate={metrics['left_win_rate']:.2%}, "
            f"right_win_rate={metrics['right_win_rate']:.2%}, "
            f"draw_rate={metrics['draw_rate']:.2%}, "
            f"avg_left_score={metrics['avg_left_score']:.2f}, "
            f"avg_right_score={metrics['avg_right_score']:.2f}, "
            f"avg_steps={metrics['avg_steps']:.1f}"
        )
        return

    if args.mode == "human_vs_agent":
        agent = create_agent(args.agent_algo, env.obs_dim, env.action_space, device=args.device)
        agent_ckpt = _resolve_checkpoint(args.model_dir, args.agent_algo, args.agent_checkpoint)
        if not agent_ckpt.exists():
            raise FileNotFoundError(f"Agent checkpoint not found: {agent_ckpt}")
        agent.load(str(agent_ckpt))

        metrics = launch_human_vs_agent(
            agent=agent,
            env=env,
            human_side=args.human_side,
            deterministic=args.deterministic,
            fps=args.fps,
            episodes=args.episodes,
        )
        print("[play] mode=human_vs_agent")
        print(
            "[play] "
            f"episodes={int(metrics.get('episodes', 0.0))}, "
            f"left_win_rate={metrics.get('left_win_rate', 0.0):.2%}, "
            f"right_win_rate={metrics.get('right_win_rate', 0.0):.2%}, "
            f"draw_rate={metrics.get('draw_rate', 0.0):.2%}"
        )
        return

    raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()

