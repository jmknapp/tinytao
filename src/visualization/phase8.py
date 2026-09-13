"""Substitution-test figures."""

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


def plot_substitute_exact(rows: list[dict[str, Any]], out: Path) -> Path:
    """Fraction still exact after each substitution, one group per source."""
    names = [r["name"] for r in rows]
    conds = [c["name"] for c in rows[0]["conditions"]]
    x = np.arange(len(names))
    width = 0.8 / max(len(conds), 1)
    colors = ["#293241", "#3d5a80", "#98c1d9", "#ee6c4d", "#e0aaff", "#8d99ae", "#2a9d8f"]
    fig, ax = plt.subplots(figsize=(max(8.0, 1.8 * len(names)), 4.2))
    for j, cond in enumerate(conds):
        vals = []
        for r in rows:
            c = r["conditions"][j]
            nt = c.get("n_touched") or 0
            if nt:
                vals.append(c["n_exact_touched"] / nt)
            else:
                vals.append(c["frac_exact"])
        ax.bar(x + (j - (len(conds) - 1) / 2) * width, vals, width, label=cond, color=colors[j % len(colors)])
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("fraction still domain-exact")
    ax.set_title("Substitution of fitted 1-D embeddings")
    ax.legend(fontsize=8, ncol=2)
    return _save(fig, out)
