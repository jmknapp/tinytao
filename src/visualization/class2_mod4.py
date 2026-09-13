"""Prospective H_CLASS2_MOD4 figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.analysis.fourier import discrete_log_table
from src.analysis.radial_quotient import fiber_id_from_k, product_hidden


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _xy_err(rows: list[dict], y_key: str, lo_key: str, hi_key: str):
    xs = np.array([r["p"] for r in rows], dtype=float)
    ys = np.array([r[y_key] for r in rows], dtype=float)
    lo = ys - np.array([r[lo_key] for r in rows], dtype=float)
    hi = np.array([r[hi_key] for r in rows], dtype=float) - ys
    return xs, ys, lo, hi


def _style(row: dict) -> dict:
    if row.get("p_mod_4") == 3:
        return {"marker": "s", "color": "#3d5a80"}
    return {"marker": "o", "color": "#ee6c4d"}


def plot_rate_vs_p(rows: list[dict], y_key: str, lo_key: str, hi_key: str, ylabel: str, title: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for src, label, fill in (("old", "old data", False), ("prospective", "prospective", True)):
        sub = [r for r in rows if r.get("source_kind") == src]
        if not sub:
            continue
        for r in sub:
            st = _style(r)
            y = r[y_key]
            ax.errorbar(
                [r["p"]],
                [y],
                yerr=[[y - r[lo_key]], [r[hi_key] - y]],
                fmt=st["marker"],
                color=st["color"],
                capsize=3,
                markersize=8 if fill else 7,
                markerfacecolor=st["color"] if fill else "white",
                markeredgecolor=st["color"],
                linestyle="none",
            )
    ax.plot([], [], "s", color="#3d5a80", label="p ≡ 3 (mod 4)")
    ax.plot([], [], "o", color="#ee6c4d", label="p ≡ 1 (mod 4)")
    ax.plot([], [], "s", color="#6c757d", markerfacecolor="white", markeredgecolor="#6c757d", label="old data (open)")
    ax.set_xlabel("p")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(-0.03, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    _save(fig, out)


def plot_winding_gcd(rows: list[dict], out: Path) -> None:
    primes = [r["p"] for r in rows]
    gcds = sorted({int(g) for r in rows for g in r.get("gcd_hist", {})})
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    bottom = np.zeros(len(primes))
    cmap = plt.get_cmap("tab10")
    for i, g in enumerate(gcds):
        vals = np.array([r.get("gcd_hist", {}).get(str(g), r.get("gcd_hist", {}).get(g, 0)) for r in rows], dtype=float)
        ax.bar([str(p) for p in primes], vals, bottom=bottom, label=f"gcd={g}", color=cmap(i % 10), edgecolor="white")
        bottom += vals
    ax.set_xlabel("p")
    ax.set_ylabel("exact networks")
    ax.set_title("Winding gcd among domain-exact nets")
    ax.legend(frameon=False, ncols=4, fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, out)


def plot_parity_consistency(rows: list[dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for r in rows:
        st = _style(r)
        y = r.get("radial_parity_consistency_gcd2")
        if y is None:
            continue
        ax.scatter(
            [r["p"]],
            [y],
            marker=st["marker"],
            c=st["color"],
            s=70,
            facecolors=st["color"] if r.get("source_kind") == "prospective" else "white",
            edgecolors=st["color"],
            zorder=3,
        )
    ax.set_xlabel("p")
    ax.set_ylabel("frac gcd=2 nets with global dlog-parity radius")
    ax.set_title("Radius parity consistency among gcd=2 exact nets")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    _save(fig, out)


def plot_class2_vs_mod4(rows: list[dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    for r in rows:
        x = 0 if r["p_mod_4"] == 3 else 1
        jitter = (hash(str(r["p"])) % 7 - 3) * 0.03
        st = _style(r)
        ax.scatter(
            [x + jitter],
            [r["p_class2"]],
            marker=st["marker"],
            s=80,
            c=st["color"],
            facecolors=st["color"] if r.get("source_kind") == "prospective" else "white",
            edgecolors=st["color"],
            zorder=3,
        )
        ax.annotate(f"p={r['p']}", (x + jitter, r["p_class2"]), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["p ≡ 3 (mod 4)\nC_n ≅ C_m × C2", "p ≡ 1 (mod 4)\nnot a direct product"])
    ax.set_ylabel("P(Class-2)")
    ax.set_title("Class-2 frequency vs the binary algebraic predictor")
    ax.set_ylim(-0.03, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, out)


def plot_embedding_plane(xy: np.ndarray, p: int, k: int, out: Path, title: str) -> None:
    dlog = discrete_log_table(p)
    dlog_v = np.array([int(dlog[a]) for a in range(1, p)])
    fid = fiber_id_from_k(k, dlog_v, p - 1)
    fig, ax = plt.subplots(figsize=(5.0, 4.8))
    cmap = plt.get_cmap("tab10")
    for f in np.unique(fid):
        m = fid == f
        ax.scatter(xy[m, 0], xy[m, 1], c=[cmap(int(f) % 10)], s=55, zorder=3, label=f"fiber {int(f)}")
        for i in np.where(m)[0]:
            ax.annotate(str(i + 1), (xy[i, 0], xy[i, 1]), textcoords="offset points", xytext=(3, 3), fontsize=7)
    ax.axhline(0, color="#adb5bd", lw=0.8)
    ax.axvline(0, color="#adb5bd", lw=0.8)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("Re z(a)")
    ax.set_ylabel("Im z(a)")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=7, ncols=2)
    _save(fig, out)


def plot_shell_diagram(xy: np.ndarray, p: int, out: Path, title: str) -> None:
    dlog = discrete_log_table(p)
    dlog_v = np.array([int(dlog[a]) for a in range(1, p)])
    _, pr, pth = product_hidden(xy)
    aa, bb = np.meshgrid(np.arange(1, p), np.arange(1, p), indexing="ij")
    tgt = (aa * bb) % p
    odd = (dlog_v[tgt - 1] % 2) == 1
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.scatter(pth[~odd].ravel(), pr[~odd].ravel(), s=8, c="#3d5a80", label="even dlog(ab)", alpha=0.7)
    ax.scatter(pth[odd].ravel(), pr[odd].ravel(), s=8, c="#ee6c4d", label="odd dlog(ab)", alpha=0.7)
    ax.set_xlabel("product phase (rad)")
    ax.set_ylabel("product radius |z(a)z(b)|")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.3)
    _save(fig, out)
