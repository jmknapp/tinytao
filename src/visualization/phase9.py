"""Modular-addition control figures."""

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


def plot_family_compare(add_rows: list[dict[str, Any]], mul_rows: list[dict[str, Any]], out: Path) -> Path:
    """Share of sides won by dlog vs residue-Fourier, addition vs multiplication."""
    names = [r["name"] for r in add_rows]
    x = np.arange(len(names))
    width = 0.18
    fig, ax = plt.subplots(figsize=(max(7.0, 1.5 * len(names)), 4.2))

    def _frac(rows, key, family):
        out = []
        for r in rows:
            n = max(r[key]["n"], 1)
            out.append(r[key][family] / n)
        return out

    ax.bar(x - 1.5 * width, _frac(add_rows, "votes", "fourier"), width, color="#3d5a80", label="add Fourier")
    ax.bar(x - 0.5 * width, _frac(add_rows, "votes", "dlog"), width, color="#98c1d9", label="add dlog")
    ax.bar(x + 0.5 * width, _frac(mul_rows, "votes", "fourier"), width, color="#ee6c4d", label="mul Fourier")
    ax.bar(x + 1.5 * width, _frac(mul_rows, "votes", "dlog"), width, color="#e0aaff", label="mul dlog")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("fraction of sides won")
    ax.set_title("Embedding family vs task (same architecture)")
    ax.legend(fontsize=8, ncol=2)
    return _save(fig, out)


def plot_add_substitute(rows: list[dict[str, Any]], out: Path) -> Path:
    names = [r["name"] for r in rows]
    conds = [c["name"] for c in rows[0]["substitution"]]
    x = np.arange(len(names))
    width = 0.8 / max(len(conds), 1)
    colors = ["#293241", "#3d5a80", "#98c1d9", "#ee6c4d", "#e0aaff", "#2a9d8f"]
    fig, ax = plt.subplots(figsize=(max(8.0, 1.8 * len(names)), 4.2))
    for j, cond in enumerate(conds):
        vals = []
        for r in rows:
            c = r["substitution"][j]
            nt = c.get("n_touched") or 0
            vals.append(c["n_exact_touched"] / nt if nt else c["frac_exact"])
        ax.bar(x + (j - (len(conds) - 1) / 2) * width, vals, width, label=cond, color=colors[j % len(colors)])
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("fraction still exact (touched nets)")
    ax.set_title("Addition: substitute fitted embeddings")
    ax.legend(fontsize=8, ncol=2)
    return _save(fig, out)
