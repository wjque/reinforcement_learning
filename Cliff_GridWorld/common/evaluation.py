from __future__ import annotations

from typing import Dict


def evaluate_agent(agent, env, episodes: int = 20, deterministic: bool = True, seed: int = 0) -> Dict[str, float]:
    returns = []
    step_counts = []
    successes = 0
    cliff_fails = 0

    for ep in range(episodes):
        state = env.reset(seed=seed + ep)
        done = False
        episode_return = 0.0
        steps = 0
        event = "timeout"

        while not done and steps < env.max_steps:
            action = agent.act(state, deterministic=deterministic)
            state, reward, done, info = env.step(action)
            episode_return += reward
            steps += 1
            event = str(info.get("event", "step"))

        returns.append(episode_return)
        step_counts.append(steps)
        if event == "goal":
            successes += 1
        if event == "cliff":
            cliff_fails += 1

    eps = max(1, episodes)
    return {
        "avg_return": float(sum(returns) / eps),
        "success_rate": float(successes / eps),
        "avg_steps": float(sum(step_counts) / eps),
        "cliff_rate": float(cliff_fails / eps),
    }

