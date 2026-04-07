from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ACTION_SYMBOLS = {
    0: "^",
    1: "v",
    2: "<",
    3: ">",
}


def plot_policy_heatmap(policy: np.ndarray, env, output_path: str, title: str) -> None:
    if policy.shape[0] != env.state_space:
        raise ValueError(
            f"Policy length mismatch: expected {env.state_space}, got {policy.shape[0]}."
        )

    grid = policy.reshape(env.rows, env.cols)
    fig, ax = plt.subplots(figsize=(10, 3))
    im = ax.imshow(grid, cmap="viridis", vmin=0, vmax=env.action_space - 1)
    ax.set_title(title)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_xticks(range(env.cols))
    ax.set_yticks(range(env.rows))
    ax.grid(which="major", color="white", linestyle="-", linewidth=0.5, alpha=0.4)

    for r in range(env.rows):
        for c in range(env.cols):
            state = env.pos_to_state(r, c)
            if (r, c) in env.cliff_cells:
                label = "C"
            elif (r, c) == (env.start.row, env.start.col):
                label = "S"
            elif (r, c) == (env.goal.row, env.goal.col):
                label = "G"
            else:
                label = ACTION_SYMBOLS.get(int(policy[state]), "?")
            ax.text(c, r, label, ha="center", va="center", color="white", fontsize=10)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Action ID")
    fig.tight_layout()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
