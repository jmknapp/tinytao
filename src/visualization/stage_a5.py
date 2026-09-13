"""Stage A.5 figures: non-unit exact embeddings, fibers, products, readout."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.analysis.fourier import discrete_log_table
from src.analysis.radial_quotient import fiber_id_from_k, fiber_members, product_hidden


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_census_bars(counts: dict, out: Path, title: str) -> None:
    labels = list(counts.keys())
    vals = [counts[k] for k in labels]
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ax.bar(labels, vals, color="#3d5a80", edgecolor="white")
    ax.set_ylabel("exact networks")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, out)


def plot_embedding_plane(
    xy: np.ndarray,
    p: int,
    k: int,
    out: Path,
    title: str,
) -> None:
    dlog = discrete_log_table(p)
    dlog_vals = np.array([int(dlog[a]) for a in range(1, p)])
    n = p - 1
    fid = fiber_id_from_k(k, dlog_vals, n)
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    cmap = plt.get_cmap("tab10")
    for f in np.unique(fid):
        m = fid == f
        ax.scatter(xy[m, 0], xy[m, 1], c=[cmap(int(f) % 10)], s=70, zorder=3, label=f"fiber {int(f)}")
        for i in np.where(m)[0]:
            a = i + 1
            ax.annotate(f"{a}", (xy[i, 0], xy[i, 1]), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.axhline(0, color="#adb5bd", lw=0.8)
    ax.axvline(0, color="#adb5bd", lw=0.8)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("Re z(a)")
    ax.set_ylabel("Im z(a)")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    _save(fig, out)


def plot_polar_r_theta(xy: np.ndarray, p: int, k: int, out: Path, title: str) -> None:
    dlog = discrete_log_table(p)
    dlog_vals = np.array([int(dlog[a]) for a in range(1, p)])
    fid = fiber_id_from_k(k, dlog_vals, p - 1)
    r = np.linalg.norm(xy, axis=-1)
    th = np.arctan2(xy[:, 1], xy[:, 0])
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    cmap = plt.get_cmap("tab10")
    for f in np.unique(fid):
        m = fid == f
        ax.scatter(th[m], r[m], c=[cmap(int(f) % 10)], s=70, label=f"fiber {int(f)}")
        for i in np.where(m)[0]:
            ax.annotate(str(i + 1), (th[i], r[i]), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xlabel("learned phase θ(a) (rad)")
    ax.set_ylabel("radius |z(a)|")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.3)
    _save(fig, out)


def plot_radius_vs_dlog(xy: np.ndarray, p: int, k: int, out: Path, title: str) -> None:
    dlog = discrete_log_table(p)
    dlog_vals = np.array([int(dlog[a]) for a in range(1, p)])
    r = np.linalg.norm(xy, axis=-1)
    fid = fiber_id_from_k(k, dlog_vals, p - 1)
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    cmap = plt.get_cmap("tab10")
    for f in np.unique(fid):
        m = fid == f
        ax.scatter(dlog_vals[m], r[m], c=[cmap(int(f) % 10)], s=70, label=f"fiber {int(f)}")
        for i in np.where(m)[0]:
            ax.annotate(str(i + 1), (dlog_vals[i], r[i]), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xlabel(f"discrete log base g")
    ax.set_ylabel("radius |z(a)|")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.3)
    _save(fig, out)


def plot_radius_by_fiber(xy: np.ndarray, p: int, k: int, out: Path, title: str) -> None:
    fibers = fiber_members(k, p)
    r = np.linalg.norm(xy, axis=-1)
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    xs, hs, ls = [], [], []
    for f, members in fibers.items():
        rr = [r[a - 1] for a in members]
        xs.append(f)
        ls.append(min(rr))
        hs.append(max(rr))
        ax.scatter([f] * len(members), rr, c="#3d5a80", s=50, zorder=3)
        for a, rv in zip(members, rr):
            ax.annotate(str(a), (f, rv), textcoords="offset points", xytext=(5, 2), fontsize=8)
    ax.plot(xs, ls, "o--", color="#98c1d9", label="min in fiber")
    ax.plot(xs, hs, "s-", color="#ee6c4d", label="max in fiber")
    ax.set_xlabel("phase fiber")
    ax.set_ylabel("radius |z(a)|")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.3)
    _save(fig, out)


def plot_phase_vs_ideal(xy: np.ndarray, p: int, k: int, phi: float, out: Path, title: str) -> None:
    dlog = discrete_log_table(p)
    n = p - 1
    dlog_vals = np.array([int(dlog[a]) for a in range(1, p)], dtype=np.float64)
    th = np.arctan2(xy[:, 1], xy[:, 0])
    ideal = phi + 2.0 * np.pi * k * dlog_vals / n
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    ax.scatter(ideal, th, c="#3d5a80", s=60)
    for i in range(n):
        ax.annotate(str(i + 1), (ideal[i], th[i]), textcoords="offset points", xytext=(4, 4), fontsize=8)
    lim = [min(ideal.min(), th.min()) - 0.3, max(ideal.max(), th.max()) + 0.3]
    ax.plot(lim, lim, color="#adb5bd", lw=1)
    ax.set_xlabel("ideal non-unit phase φ + 2π k log_g(a)/(p−1)")
    ax.set_ylabel("learned phase")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    _save(fig, out)


def plot_product_plane(xy: np.ndarray, W: np.ndarray, p: int, out: Path, title: str) -> None:
    h, _, _ = product_hidden(xy)
    n = p - 1
    target = ((np.arange(1, p)[:, None] * np.arange(1, p)[None, :]) % p).reshape(-1)
    hh = h.reshape(-1, 2)
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    sc = ax.scatter(hh[:, 0], hh[:, 1], c=target, cmap="tab20", s=18, alpha=0.85)
    w = np.asarray(W)
    if w.shape[0] == 2:
        wxy = w.T
    else:
        wxy = w.reshape(2, n).T
    scale = np.linalg.norm(hh, axis=1).mean()
    wxy = wxy / (np.linalg.norm(wxy, axis=1, keepdims=True) + 1e-12) * scale
    for c in range(n):
        ax.arrow(0, 0, wxy[c, 0], wxy[c, 1], head_width=0.08 * scale, color="#293241", alpha=0.7, length_includes_head=True)
        ax.annotate(str(c + 1), (wxy[c, 0], wxy[c, 1]), fontsize=7, color="#293241")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("Re h(a,b)")
    ax.set_ylabel("Im h(a,b)")
    ax.set_title(title)
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="target residue ab")
    _save(fig, out)


def plot_decision_regions(W: np.ndarray, b: np.ndarray, xy: np.ndarray, p: int, out: Path, title: str) -> None:
    n = p - 1
    w = np.asarray(W, dtype=np.float64)
    if w.shape[0] == 2:
        wxy = w.T
    else:
        wxy = w.reshape(2, n).T
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    h, _, _ = product_hidden(xy)
    hh = h.reshape(-1, 2)
    pad = 0.15 * (hh.max() - hh.min() + 1e-6)
    xmin, xmax = hh[:, 0].min() - pad, hh[:, 0].max() + pad
    ymin, ymax = hh[:, 1].min() - pad, hh[:, 1].max() + pad
    g = np.linspace
    xx, yy = np.meshgrid(g(xmin, xmax, 250), g(ymin, ymax, 250))
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1)
    pred = (grid @ wxy.T + bb).argmax(axis=1).reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.contourf(xx, yy, pred, levels=np.arange(-0.5, n + 0.5, 1), cmap="tab20", alpha=0.35)
    target = ((np.arange(1, p)[:, None] * np.arange(1, p)[None, :]) % p).reshape(-1)
    ax.scatter(hh[:, 0], hh[:, 1], c=target, cmap="tab20", s=14, edgecolors="k", linewidths=0.2)
    ax.set_xlabel("Re h")
    ax.set_ylabel("Im h")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    _save(fig, out)


def plot_metric_hist(values: np.ndarray, out: Path, xlabel: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    ax.hist(values, bins=24, color="#3d5a80", edgecolor="white")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("non-unit exact networks")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, out)


def plot_aligned_overlay(xys: list[np.ndarray], p: int, k: int, out: Path, title: str) -> None:
    """Overlay unit-mean-normalized embeddings after rotating mean phase to 0."""
    dlog = discrete_log_table(p)
    dlog_vals = np.array([int(dlog[a]) for a in range(1, p)])
    fid = fiber_id_from_k(k, dlog_vals, p - 1)
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    cmap = plt.get_cmap("tab10")
    for xy in xys:
        r = np.linalg.norm(xy, axis=-1, keepdims=True)
        xy_n = xy / (r.mean() + 1e-12)
        th = np.arctan2(xy_n[:, 1], xy_n[:, 0])
        mu = np.arctan2(np.mean(np.sin(th)), np.mean(np.cos(th)))
        c, s = np.cos(-mu), np.sin(-mu)
        rot = np.stack([c * xy_n[:, 0] - s * xy_n[:, 1], s * xy_n[:, 0] + c * xy_n[:, 1]], axis=1)
        for f in np.unique(fid):
            m = fid == f
            ax.scatter(rot[m, 0], rot[m, 1], c=[cmap(int(f) % 10)], s=12, alpha=0.25, linewidths=0)
    ax.axhline(0, color="#adb5bd", lw=0.8)
    ax.axvline(0, color="#adb5bd", lw=0.8)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("aligned Re z / mean r")
    ax.set_ylabel("aligned Im z / mean r")
    ax.set_title(title)
    _save(fig, out)
