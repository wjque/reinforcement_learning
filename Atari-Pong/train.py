from __future__ import annotations

import argparse
import csv
from pathlib import Path

from common.checkpoints import SUPPORTED_ALGOS, checkpoint_filename
from common.factory import create_agent
from common.visualization import moving_average, plot_learning_curve
from env.pong_env import PongEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Actor-Critic / PPO on simplified Atari-Pong.")
    parser.add_argument(
        "--algo",
        type=str,
        required=True,
        choices=SUPPORTED_ALGOS,
        help="Algorithm name.",
    )
    parser.add_argument("--total-steps", type=int, default=300_000, help="Total training environment steps.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device.")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Checkpoint directory.")
    parser.add_argument("--log-dir", type=str, default="logs", help="Training metric CSV directory.")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output image directory.")
    parser.add_argument(
        "--render",
        type=str,
        choices=("none", "pyqt"),
        default="none",
        help="Render mode during training.",
    )
    parser.add_argument("--fps", type=int, default=60, help="Render frame rate when render=pyqt.")
    return parser.parse_args()


def save_episode_metrics_csv(path: Path, episode_returns: list[float], ma_window: int = 30) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ma = moving_average(episode_returns, window=ma_window)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "return", f"moving_avg_{ma_window}"])
        for idx, value in enumerate(episode_returns):
            writer.writerow([idx + 1, float(value), float(ma[idx])])


def main() -> None:
    args = parse_args()

    env = PongEnv()
    live_renderer = None
    if args.render == "pyqt":
        from env.pyqt_renderer import PongLiveRenderer

        live_renderer = PongLiveRenderer(title=f"Training: {args.algo}", fps=args.fps)
        env.render_hook = live_renderer.update

    agent = create_agent(
        algo=args.algo,
        obs_dim=env.obs_dim,
        action_space=env.action_space,
        device=args.device,
    )

    print(
        f"[train] algo={args.algo}, total_steps={args.total_steps}, "
        f"seed={args.seed}, device={args.device}, render={args.render}"
    )
    stats = agent.train(env=env, total_steps=args.total_steps, seed=args.seed)
    print(f"[train] finished. episodes={stats.get('episodes', 0)}")

    if live_renderer is not None:
        live_renderer.close()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / checkpoint_filename(args.algo)
    agent.save(str(ckpt_path))

    episode_returns = list(stats.get("episode_returns", []))
    log_path = Path(args.log_dir) / f"{args.algo}_metrics.csv"
    save_episode_metrics_csv(log_path, episode_returns=episode_returns)

    curve_path = Path(args.output_dir) / f"{args.algo}_learning_curve.png"
    plot_learning_curve(
        episode_returns=episode_returns,
        output_path=str(curve_path),
        title=f"Pong Training Curve ({args.algo})",
        ma_window=30,
    )

    print(f"[train] checkpoint: {ckpt_path}")
    print(f"[train] metrics csv: {log_path}")
    print(f"[train] curve image: {curve_path}")


if __name__ == "__main__":
    main()

