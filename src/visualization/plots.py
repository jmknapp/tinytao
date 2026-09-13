"""Phase-1 figures. Matplotlib only; no interactive backends required."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_phase1(
    out_dir: Path,
    log_epochs: list[int],
    train_acc: torch.Tensor,
    test_acc: torch.Tensor,
    domain_acc: torch.Tensor,
    final_domain_acc: torch.Tensor,
    labels: torch.Tensor,
    n_classes: int,
) -> list[Path]:
    """Write the Phase-1 diagnostic plots. Returns saved paths."""
    out_dir = Path(out_dir)
    paths: list[Path] = []
    t = np.asarray(log_epochs)
    tr = train_acc.numpy()
    te = test_acc.numpy()
    do = domain_acc.numpy()
    fin = final_domain_acc.numpy()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(fin, bins=min(40, max(10, int(np.sqrt(len(fin))))), range=(0, 1), color="#3d5a80", edgecolor="white")
    ax.axvline(1.0 / n_classes, color="#ee6c4d", linestyle="--", label=f"chance {1/n_classes:.3f}")
    ax.set_xlabel("complete-domain accuracy")
    ax.set_ylabel("networks")
    ax.set_title("Final complete-domain accuracy")
    ax.legend()
    p = out_dir / "domain_acc_hist.png"
    _save(fig, p)
    paths.append(p)

    fig, ax = plt.subplots(figsize=(7, 4))
    for arr, name, color in (
        (tr, "train", "#3d5a80"),
        (te, "held-out", "#98c1d9"),
        (do, "complete domain", "#ee6c4d"),
    ):
        mean = arr.mean(axis=0)
        lo, hi = np.quantile(arr, [0.1, 0.9], axis=0)
        ax.plot(t, mean, label=name, color=color)
        ax.fill_between(t, lo, hi, color=color, alpha=0.18)
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.02)
    ax.set_title("Population accuracy (mean, 10–90% band)")
    ax.legend()
    p = out_dir / "accuracy_trajectories.png"
    _save(fig, p)
    paths.append(p)

    # A few individual organisms: best, median, worst, plus one sudden if any.
    order = np.argsort(fin)
    picks = [order[0], order[len(order) // 2], order[-1]]
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, name in zip(picks, ("worst", "median", "best")):
        ax.plot(t, do[i], label=f"{name} #{i} (final {fin[i]:.2f})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("complete-domain accuracy")
    ax.set_ylim(0, 1.02)
    ax.set_title("Representative learning trajectories")
    ax.legend()
    p = out_dir / "representative_trajectories.png"
    _save(fig, p)
    paths.append(p)

    first = np.full(len(fin), np.nan)
    reached = do >= (1.0 - 1e-12)
    for i in range(len(fin)):
        hits = np.where(reached[i])[0]
        if len(hits):
            first[i] = t[hits[0]]
    fig, ax = plt.subplots(figsize=(7, 4))
    vals = first[np.isfinite(first)]
    if len(vals):
        ax.hist(vals, bins=min(30, max(5, int(np.sqrt(len(vals))))), color="#293241", edgecolor="white")
        ax.set_xlabel("epoch of first exact complete-domain solve")
        ax.set_ylabel("networks")
        ax.set_title(f"Exact-solver emergence (n={len(vals)})")
    else:
        ax.text(0.5, 0.5, "no exact solvers in this run", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        ax.set_title("Exact-solver emergence")
    p = out_dir / "exact_emergence.png"
    _save(fig, p)
    paths.append(p)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    names = ["fail", "partial", "memorize", "generalize", "exact"]
    counts = [(labels.numpy() == i).sum() for i in range(5)]
    ax.bar(names, counts, color=["#ee6c4d", "#e0fbfc", "#98c1d9", "#3d5a80", "#293241"])
    ax.set_ylabel("networks")
    ax.set_title("Outcome bins")
    p = out_dir / "outcome_bins.png"
    _save(fig, p)
    paths.append(p)

    return paths
