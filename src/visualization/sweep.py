"""Width-sweep figures for Phase 3."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_width_sweep(rows: list[dict[str, Any]], out_dir: Path) -> list[Path]:
    out_dir = Path(out_dir)
    paths: list[Path] = []
    widths = np.array([r["hidden"] for r in rows], dtype=float)
    params = np.array([r["params_per_network"] for r in rows], dtype=float)
    p = np.array([r["frac_exact"] for r in rows], dtype=float)
    lo = np.array([r["frac_exact_lo"] for r in rows], dtype=float)
    hi = np.array([r["frac_exact_hi"] for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(widths, p, yerr=[p - lo, hi - p], fmt="o-", color="#293241", capsize=3)
    ax.set_xlabel("hidden width")
    ax.set_ylabel("fraction exact solvers")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Pr(exact solver) vs hidden width")
    pth = out_dir / "exact_vs_width.png"
    _save(fig, pth)
    paths.append(pth)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(params, p, yerr=[p - lo, hi - p], fmt="o-", color="#3d5a80", capsize=3)
    ax.set_xlabel("parameters per network")
    ax.set_ylabel("fraction exact solvers")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Pr(exact solver) vs parameter count")
    pth = out_dir / "exact_vs_params.png"
    _save(fig, pth)
    paths.append(pth)

    keys = ["fail", "partial", "memorize", "generalize", "exact"]
    colors = ["#ee6c4d", "#e0fbfc", "#98c1d9", "#3d5a80", "#293241"]
    stacked = np.array([[r["fractions"].get(k, 0.0) for k in keys] for r in rows])
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bottom = np.zeros(len(rows))
    x = np.arange(len(rows))
    for i, (k, c) in enumerate(zip(keys, colors)):
        ax.bar(x, stacked[:, i], bottom=bottom, color=c, label=k, width=0.8)
        bottom = bottom + stacked[:, i]
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(w)) for w in widths])
    ax.set_xlabel("hidden width")
    ax.set_ylabel("fraction of population")
    ax.set_ylim(0, 1.02)
    ax.set_title("Outcome bins vs width")
    ax.legend(ncols=5, fontsize=8)
    pth = out_dir / "outcomes_vs_width.png"
    _save(fig, pth)
    paths.append(pth)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(widths, [r["mean_domain_acc"] for r in rows], "o-", color="#ee6c4d", label="mean domain")
    ax.plot(widths, [r["max_domain_acc"] for r in rows], "s--", color="#293241", label="max domain")
    ax.plot(widths, [r["mean_train_acc"] for r in rows], "^-", color="#3d5a80", label="mean train")
    ax.set_xlabel("hidden width")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.02)
    ax.set_title("Accuracy vs width")
    ax.legend()
    pth = out_dir / "accuracy_vs_width.png"
    _save(fig, pth)
    paths.append(pth)

    emerged = [r.get("median_first_exact_epoch") for r in rows]
    if any(v is not None for v in emerged):
        fig, ax = plt.subplots(figsize=(7, 4))
        xs, ys = [], []
        for w, v in zip(widths, emerged):
            if v is not None:
                xs.append(w)
                ys.append(v)
        ax.plot(xs, ys, "o-", color="#293241")
        ax.set_xlabel("hidden width")
        ax.set_ylabel("median epoch of first exact solve")
        ax.set_title("When exact solvers appear")
        pth = out_dir / "emergence_vs_width.png"
        _save(fig, pth)
        paths.append(pth)

    return paths
