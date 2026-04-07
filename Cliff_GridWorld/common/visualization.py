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


def _grid_figsize(rows: int, cols: int) -> tuple[float, float]:
    # Keep near-square cells while bounding output size for very small/large maps.
    cell_w = 0.62
    cell_h = 0.62
    width = float(np.clip(cols * cell_w + 2.0, 6.0, 22.0))
    height = float(np.clip(rows * cell_h + 1.5, 3.5, 14.0))
    return width, height


def _cell_fontsize(rows: int, cols: int) -> float:
    # Reduce text size when the grid grows, so labels stay readable.
    scale = max(rows, cols)
    return float(np.clip(12.0 - 0.22 * scale, 5.5, 10.0))


def _set_sparse_tick_labels(ax: plt.Axes, rows: int, cols: int) -> None:
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))

    x_step = max(1, cols // 12)
    y_step = max(1, rows // 10)
    ax.set_xticklabels([str(c) if c % x_step == 0 else "" for c in range(cols)])
    ax.set_yticklabels([str(r) if r % y_step == 0 else "" for r in range(rows)])


def plot_policy_heatmap(policy: np.ndarray, env, output_path: str, title: str) -> None:
    if policy.shape[0] != env.state_space:
        raise ValueError(
            f"Policy length mismatch: expected {env.state_space}, got {policy.shape[0]}."
        )

    grid = policy.reshape(env.rows, env.cols)
    fig, ax = plt.subplots(figsize=_grid_figsize(env.rows, env.cols))
    im = ax.imshow(grid, cmap="viridis", vmin=0, vmax=env.action_space - 1)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    _set_sparse_tick_labels(ax, env.rows, env.cols)
    ax.grid(which="major", color="white", linestyle="-", linewidth=0.5, alpha=0.4)

    text_size = _cell_fontsize(env.rows, env.cols)
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
            ax.text(c, r, label, ha="center", va="center", color="white", fontsize=text_size)

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
    fig, ax = plt.subplots(figsize=_grid_figsize(env.rows, env.cols))
    ax.imshow(canvas, cmap="Greys", vmin=0, vmax=1)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    _set_sparse_tick_labels(ax, env.rows, env.cols)
    ax.grid(which="major", color="black", linestyle="-", linewidth=0.6, alpha=0.25)

    # Draw static map annotations first.
    text_size = _cell_fontsize(env.rows, env.cols)
    for r in range(env.rows):
        for c in range(env.cols):
            if (r, c) in env.cliff_cells:
                rect = plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="#ff6b6b", alpha=0.35)
                ax.add_patch(rect)
                ax.text(c, r, "C", ha="center", va="center", color="#8b0000", fontsize=text_size, fontweight="bold")
            elif (r, c) == (env.start.row, env.start.col):
                rect = plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="#74c69d", alpha=0.35)
                ax.add_patch(rect)
                ax.text(c, r, "S", ha="center", va="center", color="#1b4332", fontsize=text_size, fontweight="bold")
            elif (r, c) == (env.goal.row, env.goal.col):
                rect = plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="#4dabf7", alpha=0.35)
                ax.add_patch(rect)
                ax.text(c, r, "G", ha="center", va="center", color="#0b3d91", fontsize=text_size, fontweight="bold")

    rows = [p[0] for p in trajectory]
    cols = [p[1] for p in trajectory]
    line_width = float(np.clip(2.6 - 0.04 * max(env.rows, env.cols), 1.2, 2.2))
    marker_size = float(np.clip(6.0 - 0.08 * max(env.rows, env.cols), 2.0, 4.5))
    ax.plot(cols, rows, color="#f08c00", linewidth=line_width, marker="o", markersize=marker_size)
    ax.scatter(cols[0], rows[0], color="#2b8a3e", s=70, marker="o", zorder=3, label="Start")
    ax.scatter(cols[-1], rows[-1], color="#c92a2a", s=70, marker="X", zorder=3, label="End")

    if len(trajectory) > 1:
        for idx in range(1, len(trajectory)):
            dr = rows[idx] - rows[idx - 1]
            dc = cols[idx] - cols[idx - 1]
            if dr == 0 and dc == 0:
                continue
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
