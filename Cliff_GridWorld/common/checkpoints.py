from __future__ import annotations

TABLE_ALGOS = {"policy_iteration", "value_iteration", "sarsa", "q_learning"}
POLICY_GRAD_ALGOS = {"actor_critic", "ppo"}
SUPPORTED_ALGOS = tuple(sorted(TABLE_ALGOS | POLICY_GRAD_ALGOS))


def checkpoint_extension(algo: str) -> str:
    if algo in TABLE_ALGOS:
        return ".npz"
    if algo in POLICY_GRAD_ALGOS:
        return ".pt"
    raise ValueError(f"Unsupported algorithm '{algo}'.")


def checkpoint_filename(algo: str) -> str:
    return f"{algo}{checkpoint_extension(algo)}"

