from __future__ import annotations

def create_agent(
    algo: str,
    state_space: int,
    action_space: int,
    *,
    device: str = "cpu",
    rows: int = 4,
    cols: int = 12,
):
    if algo == "policy_iteration":
        from algorithms.value_based.policy_iteration import PolicyIterationAgent

        return PolicyIterationAgent(state_space, action_space)
    if algo == "value_iteration":
        from algorithms.value_based.value_iteration import ValueIterationAgent

        return ValueIterationAgent(state_space, action_space)
    if algo == "sarsa":
        from algorithms.value_based.sarsa import SarsaAgent

        return SarsaAgent(state_space, action_space)
    if algo == "q_learning":
        from algorithms.value_based.q_learning import QLearningAgent

        return QLearningAgent(state_space, action_space)
    if algo == "actor_critic":
        try:
            from algorithms.policy_gradient.actor_critic import ActorCriticAgent
        except ModuleNotFoundError as exc:
            raise ImportError(
                "actor_critic requires PyTorch. Install with `pip install torch`."
            ) from exc
        return ActorCriticAgent(state_space, action_space, rows=rows, cols=cols, device=device)
    if algo == "ppo":
        try:
            from algorithms.policy_gradient.ppo import PPOAgent
        except ModuleNotFoundError as exc:
            raise ImportError("ppo requires PyTorch. Install with `pip install torch`.") from exc
        return PPOAgent(state_space, action_space, rows=rows, cols=cols, device=device)
    raise ValueError(f"Unsupported algorithm '{algo}'.")
