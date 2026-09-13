"""Figures for the Phase 7 falsifiers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_orbit_bars(rows: list[dict[str, Any]], out: Path) -> Path:
    """Mean dlog R² at baseline and after three residue relabelings."""
    names = [r["name"] for r in rows]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(6.5, 1.6 * len(names)), 4.0))
    width = 0.18
    series = [
        ("base", [r["high_dlog"]["base_dlog"] for r in rows], "#293241"),
        ("× c", [r["high_dlog"]["multiplicative"]["mean_dlog"] for r in rows], "#3d5a80"),
        ("+ k", [r["high_dlog"]["additive"]["mean_dlog"] for r in rows], "#ee6c4d"),
        ("random", [r["high_dlog"]["random"]["mean_dlog"] for r in rows], "#98c1d9"),
    ]
    for i, (label, vals, color) in enumerate(series):
        ax.bar(x + (i - 1.5) * width, vals, width, label=label, color=color)
    ax.axhline(0.85, color="#8d99ae", ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("mean dlog $R^2$ (sides with base ≥ 0.85)")
    ax.set_title("Embedding family after residue relabelings")
    ax.legend()
    return _save(fig, out)


def plot_factor_bars(rows: list[dict[str, Any]], out: Path) -> Path:
    names = [r["name"] for r in rows]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(6.5, 1.6 * len(names)), 4.0))
    width = 0.18
    series = [
        ("product", [r["hidden_factor"]["product"] for r in rows], "#293241"),
        ("dlog sum", [r["hidden_factor"]["dlog_sum"] for r in rows], "#3d5a80"),
        ("sum", [r["hidden_factor"]["sum"] for r in rows], "#ee6c4d"),
        ("zero pat.", [r["hidden_factor"]["zero_pattern"] for r in rows], "#98c1d9"),
    ]
    for i, (label, vals, color) in enumerate(series):
        ax.bar(x + (i - 1.5) * width, vals, width, label=label, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("mean hidden-map factor $R^2$")
    ax.set_title("Is a hidden unit a function of one invariant?")
    ax.legend()
    return _save(fig, out)
