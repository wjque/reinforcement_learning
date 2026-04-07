# Atari-Pong (PyQt6) with Actor-Critic / PPO

This project implements a simplified Pong environment for reinforcement learning:

- Custom two-player `PongEnv` with low-dimensional observations
- Shared-policy self-play training
- `actor_critic` and `ppo` policy-gradient agents
- `agent_vs_agent` and `human_vs_agent` play modes
- Optional PyQt6 real-time rendering

## Project layout

```text
Atari-Pong/
  env/
    pong_env.py
    pyqt_renderer.py
  algorithms/
    policy_gradient/
      actor_critic.py
      ppo.py
  common/
    base_agent.py
    factory.py
    networks.py
    buffers.py
    checkpoints.py
    visualization.py
  train.py
  play.py
  tests/
```

## Environment API

- `reset(seed: int | None = None) -> tuple[np.ndarray, np.ndarray]`
- `step(left_action: int, right_action: int) -> tuple[tuple[np.ndarray, np.ndarray], tuple[float, float], bool, dict]`
- `copy() -> PongEnv`

Key constants:

- `action_space = 3` (`UP`, `STAY`, `DOWN`)
- `obs_dim = 10`
- episode ends at `target_score` or `max_steps`

## Install

```bash
pip install -r requirements.txt
```

## Train

```bash
python train.py --algo actor_critic --total-steps 300000 --device cpu --render none
python train.py --algo ppo --total-steps 300000 --device cpu --render none
```

Outputs:

- checkpoints: `checkpoints/<algo>.pt`
- metrics csv: `logs/<algo>_metrics.csv`
- curve: `outputs/<algo>_learning_curve.png`

## Agent vs Agent

```bash
python play.py --mode agent_vs_agent --left-algo ppo --right-algo actor_critic --episodes 20 --render none
```

## Human vs Agent (PyQt6)

```bash
python play.py --mode human_vs_agent --agent-algo ppo --human-side left --episodes 1 --fps 60
```

Controls:

- `W` / `S` for left paddle
- `Up` / `Down` keys are also supported

## Run tests

```bash
python -m unittest discover -s tests -p "test_*.py"
```

