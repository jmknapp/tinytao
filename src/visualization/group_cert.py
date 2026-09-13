"""Figures for Model C group-law certificate."""

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


def heatmap(mat: np.ndarray, out: Path, title: str, xlabel: str, ylabel: str, xticklabels, yticklabels) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    im = ax.imshow(mat, origin="upper", cmap="magma")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(xticklabels)))
    ax.set_xticklabels(xticklabels, fontsize=8)
    ax.set_yticks(range(len(yticklabels)))
    ax.set_yticklabels(yticklabels, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, out)


def hist_deg(values_rad: np.ndarray, out: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.hist(np.degrees(values_rad), bins=40, color="#3d5a80", edgecolor="white")
    ax.set_xlabel("centered |delta| (degrees)")
    ax.set_ylabel("networks")
    ax.set_title(title)
    _save(fig, out)
