from __future__ import annotations

from typing import Dict


def evaluate_head_to_head(
    left_agent,
    right_agent,
    env,
    episodes: int = 20,
    deterministic: bool = True,
    seed: int = 123,
) -> Dict[str, float]:
    left_wins = 0
    right_wins = 0
    draws = 0
    left_scores: list[int] = []
    right_scores: list[int] = []
    steps_list: list[int] = []

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
        steps_list.append(int(env.steps))

        if left_score > right_score:
            left_wins += 1
        elif right_score > left_score:
            right_wins += 1
        else:
            draws += 1

    eps = max(episodes, 1)
    return {
        "left_win_rate": float(left_wins / eps),
        "right_win_rate": float(right_wins / eps),
        "draw_rate": float(draws / eps),
        "avg_left_score": float(sum(left_scores) / eps),
        "avg_right_score": float(sum(right_scores) / eps),
        "avg_steps": float(sum(steps_list) / eps),
    }

