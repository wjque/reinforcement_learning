from __future__ import annotations

import argparse
from pathlib import Path

from common.checkpoints import SUPPORTED_ALGOS, checkpoint_filename
from common.factory import create_agent
from env.cliff_gridworld import CliffGridWorldEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an agent on Cliff GridWorld.")
    parser.add_argument(
        "--algo",
        type=str,
        required=True,
        choices=SUPPORTED_ALGOS,
        help="Algorithm name.",
    )
    parser.add_argument("--episodes", type=int, default=500, help="Training episodes.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Checkpoint directory.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device for policy-gradient agents.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = CliffGridWorldEnv()
    agent = create_agent(
        args.algo,
        env.state_space,
        env.action_space,
        device=args.device,
        rows=env.rows,
        cols=env.cols,
    )

    print(f"[train] algo={args.algo}, episodes={args.episodes}, seed={args.seed}")
    stats = agent.train(env, episodes=args.episodes, seed=args.seed)
    print(f"[train] done. stats keys: {list(stats.keys())}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / checkpoint_filename(args.algo)
    agent.save(str(ckpt_path))
    print(f"[train] checkpoint saved to: {ckpt_path}")


if __name__ == "__main__":
    main()

