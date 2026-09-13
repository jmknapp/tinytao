"""Stage A figures: cross-prime H=1 Model C held-row sweep."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_exact_vs_p(rows: list[dict], out: Path, x_key: str, xlabel: str, title: str) -> None:
    xs = [r[x_key] for r in rows]
    ys = [r["exact_fraction"] for r in rows]
    lo = [r["exact_fraction"] - r["wilson_lo"] for r in rows]
    hi = [r["wilson_hi"] - r["exact_fraction"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o-", color="#3d5a80", capsize=3, markersize=6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("P(exact held-row generalizer)")
    ax.set_title(title)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    _save(fig, out)


def plot_families_vs_phi(rows: list[dict], out: Path) -> None:
    xs = [r["phi"] for r in rows]
    ys = [r["n_observed_faithful_families"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.scatter(xs, ys, s=60, color="#3d5a80", zorder=3)
    for r in rows:
        ax.annotate(f"p={r['p']}", (r["phi"], r["n_observed_faithful_families"]), textcoords="offset points", xytext=(6, 4), fontsize=8)
    m = max(xs + ys + [1])
    ax.plot([0, m], [0, m], color="#98c1d9", linestyle="--", label="observed = φ(p−1)")
    ax.set_xlabel("φ(p−1) (faithful winding classes)")
    ax.set_ylabel("observed unit windings among exact nets")
    ax.set_title("Observed faithful families vs available characters")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    _save(fig, out)


def plot_hom_vs_p(rows: list[dict], out: Path) -> None:
    xs = [r["p"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.plot(xs, [r["mean_hom_deg"] for r in rows], "o-", color="#3d5a80", label="full domain")
    ax.plot(xs, [r["held_row_hom_deg"] for r in rows], "s--", color="#ee6c4d", label="held row")
    ax.set_xlabel("p")
    ax.set_ylabel("mean |δ_c| (degrees)")
    ax.set_title("Centered homomorphism error among exact generalizers")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    _save(fig, out)


def plot_symbolic_vs_p(rows: list[dict], out: Path) -> None:
    xs = [r["p"] for r in rows]
    ys = [r["symbolic_exact_fraction"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.plot(xs, ys, "o-", color="#3d5a80")
    ax.set_xlabel("p")
    ax.set_ylabel("P(symbolically replaceable | exact)")
    ax.set_title("Exact roots-of-unity substitution")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.3)
    _save(fig, out)


def plot_table_vs_dim(rows: list[dict], out: Path) -> None:
    xs = [r["p"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6))
    axes[0].plot(xs, [r["n_domain"] for r in rows], "o-", color="#3d5a80")
    axes[0].set_xlabel("p")
    axes[0].set_ylabel("table size (p−1)²")
    axes[0].set_title("Operation table grows")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(xs, [2] * len(rows), "o-", color="#ee6c4d")
    axes[1].set_xlabel("p")
    axes[1].set_ylabel("embedding dimension")
    axes[1].set_ylim(0, 4)
    axes[1].set_title("Latent dim stays 2")
    axes[1].grid(True, alpha=0.3)
    fig.suptitle("Table size vs constant 2-D representation", fontsize=11)
    _save(fig, out)


def plot_params_vs_p(rows: list[dict], out: Path) -> None:
    xs = [r["p"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.plot(xs, [r["params"] for r in rows], "o-", color="#3d5a80", label="total params = 5(p−1)")
    ax.plot(xs, [r["params_per_element"] for r in rows], "s--", color="#ee6c4d", label="params / (p−1)")
    ax.set_xlabel("p")
    ax.set_ylabel("parameter count")
    ax.set_title("Total parameters grow; latent dim does not")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    _save(fig, out)


def plot_winding_histograms(rows: list[dict], out: Path) -> None:
    n = len(rows)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.4), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, r in zip(axes, rows):
        table = r["winding_table"]
        ks = [t["k"] for t in table]
        counts = [t["count"] for t in table]
        colors = ["#3d5a80" if t["unit"] else "#adb5bd" for t in table]
        ax.bar(ks, counts, color=colors, edgecolor="white")
        ax.set_xlabel("k")
        ax.set_title(f"p={r['p']}, n={r['n']}")
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("exact generalizers")
    fig.suptitle("Winding frequency (blue = unit mod p−1)", fontsize=11)
    _save(fig, out)
