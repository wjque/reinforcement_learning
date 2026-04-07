from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

from common.checkpoints import SUPPORTED_ALGOS, checkpoint_filename
from common.evaluation import evaluate_agent
from common.factory import create_agent
from common.visualization import plot_policy_heatmap, plot_trajectory
from env.cliff_gridworld import CliffGridWorldEnv


def collect_episode_trajectory(agent, env, deterministic: bool, seed: int) -> tuple[list[tuple[int, int]], str]:
    """Roll out one episode and collect visited grid positions."""
    state = env.reset(seed=seed)
    row, col = env.state_to_pos(state)
    trajectory = [(row, col)]
    done = False
    end_event = "timeout"
    steps = 0

    while not done and steps < env.max_steps:
        action = agent.act(state, deterministic=deterministic)
        state, _reward, done, info = env.step(action)
        row, col = env.state_to_pos(state)
        trajectory.append((row, col))
        end_event = str(info.get("event", "step"))
        steps += 1

    return trajectory, end_event


def run_play(
    algos: list[str],
    model_dir: str,
    output_dir: str,
    episodes: int,
    deterministic: bool,
    seed: int = 123,
    device: str = "cpu",
) -> Dict[str, Dict[str, float]]:
    model_path = Path(model_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Dict[str, float]] = {}
    base_env = CliffGridWorldEnv(
            rows=10,
            cols=20,
            start=(3, 0),
            goal=(3, 15),
        )
    for algo in algos:
        env = base_env.copy()
        agent = create_agent(
            algo,
            env.state_space,
            env.action_space,
            device=device,
            rows=env.rows,
            cols=env.cols,
        )
        ckpt = model_path / checkpoint_filename(algo)
        if not ckpt.exists():
            raise FileNotFoundError(
                f"Checkpoint not found for '{algo}': {ckpt}. "
                f"Please run train.py first, e.g. `python train.py --algo {algo}`."
            )
        agent.load(str(ckpt))

        metrics = evaluate_agent(
            agent=agent,
            env=env,
            episodes=episodes,
            deterministic=deterministic,
            seed=seed,
        )
        results[algo] = metrics

        heatmap_path = out_path / f"policy_{algo}.png"
        plot_policy_heatmap(
            policy=agent.get_policy(),
            env=env,
            output_path=str(heatmap_path),
            title=f"Policy Heatmap: {algo}",
        )

        trajectory_env = base_env.copy()
        trajectory, end_event = collect_episode_trajectory(
            agent=agent,
            env=trajectory_env,
            deterministic=deterministic,
            seed=seed,
        )
        trajectory_path = out_path / f"trajectory_{algo}.png"
        plot_trajectory(
            trajectory=trajectory,
            env=trajectory_env,
            output_path=str(trajectory_path),
            title=f"Trajectory: {algo}",
            end_event=end_event,
        )

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained Cliff GridWorld agents.")
    parser.add_argument(
        "--algos",
        nargs="+",
        default=list(SUPPORTED_ALGOS),
        choices=SUPPORTED_ALGOS,
        help="Algorithms to evaluate.",
    )
    parser.add_argument("--model-dir", type=str, default="checkpoints", help="Checkpoint directory.")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory.")
    parser.add_argument("--episodes", type=int, default=20, help="Evaluation episodes.")
    parser.add_argument("--seed", type=int, default=123, help="Evaluation seed.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device for policy-gradient agents.")
    parser.add_argument(
        "--deterministic",
        dest="deterministic",
        action="store_true",
        default=True,
        help="Use deterministic action selection (default).",
    )
    parser.add_argument(
        "--stochastic",
        dest="deterministic",
        action="store_false",
        help="Use stochastic action selection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_play(
        algos=args.algos,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        episodes=args.episodes,
        deterministic=args.deterministic,
        seed=args.seed,
        device=args.device,
    )

    print("[play] evaluation summary")
    for algo, metrics in results.items():
        print(
            f"  - {algo:15s} "
            f"avg_return={metrics['avg_return']:.2f}, "
            f"success_rate={metrics['success_rate']:.2%}, "
            f"avg_steps={metrics['avg_steps']:.2f}, "
            f"cliff_rate={metrics['cliff_rate']:.2%}"
        )


if __name__ == "__main__":
    main()
