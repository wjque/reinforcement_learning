from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def moving_average(values: Sequence[float], window: int = 30) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return arr
    if window <= 1:
        return arr
    out = np.zeros_like(arr)
    for idx in range(arr.size):
        start = max(0, idx - window + 1)
        out[idx] = float(arr[start : idx + 1].mean())
    return out


def plot_learning_curve(
    episode_returns: Sequence[float],
    output_path: str,
    title: str = "Training Learning Curve",
    ma_window: int = 30,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    returns = np.asarray(episode_returns, dtype=np.float32)
    steps = np.arange(1, len(returns) + 1)
    smoothed = moving_average(returns, window=ma_window)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    if len(returns) > 0:
        ax.plot(steps, returns, color="#4c6ef5", alpha=0.35, linewidth=1.0, label="episode return")
        ax.plot(steps, smoothed, color="#1c7ed6", linewidth=2.0, label=f"moving avg ({ma_window})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)

