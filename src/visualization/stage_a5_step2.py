"""Step 2 figures: causal masks and substitution exactness."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def heatmap_bool(mat: np.ndarray, out: Path, title: str, xlabel: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.imshow(mat.astype(float), origin="upper", cmap="magma", vmin=0, vmax=1)
    n = mat.shape[0]
    ax.set_xticks(range(n))
    ax.set_xticklabels([str(i + 1) for i in range(n)], fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels([str(i + 1) for i in range(n)], fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    _save(fig, out)


def hist_values(vals: np.ndarray, out: Path, xlabel: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    ax.hist(vals, bins=20, color="#3d5a80", edgecolor="white")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("networks × fibers" if "F1" in xlabel or "agreement" in xlabel.lower() else "networks")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, out)


def grouped_exact_bar(labels: list[str], counts: list[int], total: int, out: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.bar(labels, counts, color="#3d5a80", edgecolor="white")
    ax.axhline(total, color="#ee6c4d", ls="--", lw=1, label=f"all {total}")
    ax.set_ylabel("domain-exact networks")
    ax.set_title(title)
    ax.set_ylim(0, total + 5)
    ax.legend(frameon=False)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, out)
