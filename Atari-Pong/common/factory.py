from __future__ import annotations


def create_agent(
    algo: str,
    obs_dim: int,
    action_space: int,
    *,
    device: str = "cpu",
):
    if algo == "actor_critic":
        try:
            from algorithms.policy_gradient.actor_critic import ActorCriticAgent
        except ModuleNotFoundError as exc:
            raise ImportError(
                "actor_critic requires PyTorch. Install with `pip install torch`."
            ) from exc
        return ActorCriticAgent(obs_dim, action_space, device=device)
    if algo == "ppo":
        try:
            from algorithms.policy_gradient.ppo import PPOAgent
        except ModuleNotFoundError as exc:
            raise ImportError("ppo requires PyTorch. Install with `pip install torch`.") from exc
        return PPOAgent(obs_dim, action_space, device=device)
    raise ValueError(f"Unsupported algorithm '{algo}'.")

