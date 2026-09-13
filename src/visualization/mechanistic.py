"""Heatmaps, embeddings, and cluster figures for Phase 4–5."""

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


def plot_unit_heatmaps(maps: torch.Tensor, out: Path, title: str, max_units: int = 8) -> Path:
    """maps [H, p, p] for one network."""
    h, p, _ = maps.shape
    h = min(h, max_units)
    cols = min(4, h)
    rows = int(np.ceil(h / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.8 * rows), squeeze=False)
    data = maps.cpu().numpy()
    vmax = np.percentile(np.abs(data[:h]), 99) + 1e-9
    for i in range(rows * cols):
        ax = axes[i // cols][i % cols]
        if i < h:
            im = ax.imshow(data[i], origin="lower", cmap="magma", vmin=0, vmax=vmax)
            ax.set_title(f"unit {i}")
            ax.set_xlabel("b")
            ax.set_ylabel("a")
            fig.colorbar(im, ax=ax, fraction=0.046)
        else:
            ax.axis("off")
    fig.suptitle(title)
    _save(fig, out)
    return out


def plot_embeddings_1d(u: torch.Tensor, v: torch.Tensor, out: Path, title: str) -> Path:
    """u,v [H, p] for one network."""
    h, p = u.shape
    fig, axes = plt.subplots(h, 1, figsize=(7, 1.6 * h), squeeze=False)
    uu, vv = u.cpu().numpy(), v.cpu().numpy()
    t = np.arange(p)
    for i in range(h):
        ax = axes[i][0]
        ax.plot(t, uu[i], "o-", label="u(a)", color="#3d5a80")
        ax.plot(t, vv[i], "s--", label="v(b)", color="#ee6c4d")
        ax.set_ylabel(f"unit {i}")
        ax.set_xticks(t)
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)
            ax.set_title(title)
    axes[-1][0].set_xlabel("residue")
    _save(fig, out)
    return out


def plot_pca(xy: np.ndarray, labels: np.ndarray, out: Path, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(6, 5))
    for k in sorted(set(labels.tolist())):
        m = labels == k
        ax.scatter(xy[m, 0], xy[m, 1], s=18, label=f"cluster {k} (n={m.sum()})")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.legend(fontsize=8)
    _save(fig, out)
    return out


def plot_distance_heatmap(dist: np.ndarray, labels: np.ndarray, out: Path, title: str) -> Path:
    order = np.argsort(labels)
    d = dist[order][:, order]
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(d, cmap="viridis", vmin=0, vmax=max(0.5, float(d.max())))
    ax.set_title(title)
    ax.set_xlabel("networks (sorted by cluster)")
    ax.set_ylabel("networks")
    fig.colorbar(im, ax=ax, fraction=0.046, label="1 − aligned |corr|")
    _save(fig, out)
    return out


def plot_r2_bars(rows: list[dict], out: Path, title: str) -> Path:
    """rows: per-unit dicts with u/v fourier_r2, onehot_r2, dlog_r2."""
    labels = [f"u{i}" for i in range(len(rows))] + [f"v{i}" for i in range(len(rows))]
    fourier = [r["u"]["fourier_r2"] for r in rows] + [r["v"]["fourier_r2"] for r in rows]
    onehot = [r["u"]["onehot_r2"] for r in rows] + [r["v"]["onehot_r2"] for r in rows]
    dlog = [r["u"]["dlog_r2"] for r in rows] + [r["v"]["dlog_r2"] for r in rows]
    x = np.arange(len(labels))
    w = 0.25
    fig, ax = plt.subplots(figsize=(max(7, 0.45 * len(labels)), 4))
    ax.bar(x - w, fourier, w, label="sinusoid", color="#293241")
    ax.bar(x, onehot, w, label="one-hot spike", color="#3d5a80")
    ax.bar(x + w, dlog, w, label="dlog sinusoid", color="#ee6c4d")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("R²")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend()
    _save(fig, out)
    return out
