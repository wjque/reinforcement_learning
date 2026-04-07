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


def plot_trajectory(
    trajectory: list[tuple[int, int]],
    env,
    output_path: str,
    title: str,
    end_event: str | None = None,
) -> None:
    """Plot one episode trajectory on top of the Cliff GridWorld map."""
    if len(trajectory) == 0:
        raise ValueError("trajectory must contain at least one grid position.")

    canvas = np.zeros((env.rows, env.cols), dtype=np.float32)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.imshow(canvas, cmap="Greys", vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_xticks(range(env.cols))
    ax.set_yticks(range(env.rows))
    ax.grid(which="major", color="black", linestyle="-", linewidth=0.6, alpha=0.25)

    # Draw static map annotations first.
    for r in range(env.rows):
        for c in range(env.cols):
            if (r, c) in env.cliff_cells:
                rect = plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="#ff6b6b", alpha=0.35)
                ax.add_patch(rect)
                ax.text(c, r, "C", ha="center", va="center", color="#8b0000", fontsize=9, fontweight="bold")
            elif (r, c) == (env.start.row, env.start.col):
                rect = plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="#74c69d", alpha=0.35)
                ax.add_patch(rect)
                ax.text(c, r, "S", ha="center", va="center", color="#1b4332", fontsize=9, fontweight="bold")
            elif (r, c) == (env.goal.row, env.goal.col):
                rect = plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="#4dabf7", alpha=0.35)
                ax.add_patch(rect)
                ax.text(c, r, "G", ha="center", va="center", color="#0b3d91", fontsize=9, fontweight="bold")

    rows = [p[0] for p in trajectory]
    cols = [p[1] for p in trajectory]
    ax.plot(cols, rows, color="#f08c00", linewidth=2.0, marker="o", markersize=4)
    ax.scatter(cols[0], rows[0], color="#2b8a3e", s=70, marker="o", zorder=3, label="Start")
    ax.scatter(cols[-1], rows[-1], color="#c92a2a", s=70, marker="X", zorder=3, label="End")

    if len(trajectory) > 1:
        for idx in range(1, len(trajectory)):
            dr = rows[idx] - rows[idx - 1]
            dc = cols[idx] - cols[idx - 1]
            ax.arrow(
                cols[idx - 1],
                rows[idx - 1],
                dc * 0.65,
                dr * 0.65,
                head_width=0.12,
                head_length=0.15,
                fc="#f08c00",
                ec="#f08c00",
                alpha=0.65,
                length_includes_head=True,
            )

    subtitle = f"steps={max(0, len(trajectory) - 1)}"
    if end_event is not None:
        subtitle += f", end={end_event}"
    ax.text(
        0.01,
        1.04,
        subtitle,
        transform=ax.transAxes,
        fontsize=9,
        color="#495057",
    )
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
