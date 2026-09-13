"""Phase 6 holdout and shrinking figures."""

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


def plot_holdouts(rows: list[dict[str, Any]], out: Path) -> Path:
    names = [r["split"] for r in rows]
    exact = [r["frac_exact"] for r in rows]
    lo = [r["frac_exact_lo"] for r in rows]
    hi = [r["frac_exact_hi"] for r in rows]
    train = [r["mean_train_acc"] for r in rows]
    test = [r["mean_test_acc"] for r in rows]
    domain = [r["mean_domain_acc"] for r in rows]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(8, 1.3 * len(names)), 4.2))
    ax.errorbar(x, exact, yerr=[np.array(exact) - lo, np.array(hi) - exact], fmt="o", color="#293241", label="Pr(exact)", capsize=3)
    ax.plot(x, train, "s--", color="#3d5a80", label="mean train")
    ax.plot(x, test, "^--", color="#98c1d9", label="mean held-out")
    ax.plot(x, domain, "d--", color="#ee6c4d", label="mean domain")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylim(-0.02, 1.05)
    ax.set_ylabel("accuracy / fraction")
    ax.set_title("Structured holdouts (width 16, same optimizer)")
    ax.legend()
    _save(fig, out)
    return out


def plot_quantize(bits: list[int], n_exact: list[int], n: int, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(bits, [k / n for k in n_exact], "o-", color="#293241")
    ax.set_xlabel("quantization bits (per-network symmetric)")
    ax.set_ylabel("fraction still exact")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(f"Width-4 exact solvers after quantization (n={n})")
    _save(fig, out)
    return out
