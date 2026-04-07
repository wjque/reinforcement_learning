# Cliff GridWorld Teaching Scaffold

This project provides a single Cliff GridWorld environment and skeletons for:

- Dynamic Programming: `policy_iteration`, `value_iteration`
- TD methods: `sarsa`, `q_learning`
- Policy gradient methods: `actor_critic`, `ppo`

The environment is fixed to:

- Grid: `4 x 12`
- Start: `(3, 0)`
- Goal: `(3, 11)`
- Cliff: `(3,1) ... (3,10)`
- Actions: up/down/left/right
- Rewards: goal `+100`, cliff `-100`, step `-1`
- Cliff behavior: stepping into cliff ends the episode (`done=True`)

## Project layout

```text
Cliff_GridWorld/
  env/
  algorithms/
    value_based/
    policy_gradient/
  common/
  train.py
  play.py
  tests/
```

## Educational TODO markers

Key update formulas are intentionally left as:

`# TODO(student): ...`

You should implement these parts yourself to learn the core math.

## Train

```bash
python train.py --algo sarsa --episodes 500 --seed 42
```

## Play (shared testing entry)

```bash
python play.py --algos sarsa q_learning --episodes 20 --deterministic
```

`play.py` expects model checkpoints under `checkpoints/` by default and outputs policy heatmaps under `outputs/`.

## Run tests

```bash
python -m unittest discover -s tests -p "test_*.py"
```

