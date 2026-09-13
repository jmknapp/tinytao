"""A/B/C comparison figure."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_model_exact(
    rows: list[dict[str, Any]],
    out: Path,
    title: str,
    model_order: list[str] | None = None,
    model_labels: dict[str, str] | None = None,
) -> Path:
    protocols = []
    for r in rows:
        if r["protocol"] not in protocols:
            protocols.append(r["protocol"])
    if model_order is None:
        model_order = []
        for r in rows:
            if r["interaction"] not in model_order:
                model_order.append(r["interaction"])
    if model_labels is None:
        model_labels = {m: m for m in model_order}
    colors = ["#293241", "#ee6c4d", "#3d5a80", "#98c1d9", "#293241"]
    n_m = len(model_order)
    x = np.arange(len(protocols))
    width = min(0.8 / n_m, 0.25)
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    for i, m in enumerate(model_order):
        vals, lo, hi = [], [], []
        for proto in protocols:
            hit = next((r for r in rows if r["protocol"] == proto and r["interaction"] == m), None)
            if hit is None:
                vals.append(0.0)
                lo.append(0.0)
                hi.append(0.0)
                continue
            vals.append(hit["frac_exact"])
            lo.append(hit["frac_exact_lo"])
            hi.append(hit["frac_exact_hi"])
        offset = (i - (n_m - 1) / 2) * width
        ax.bar(x + offset, vals, width, color=colors[i % len(colors)], label=model_labels[m])
        ax.errorbar(
            x + offset,
            vals,
            yerr=[np.array(vals) - np.array(lo), np.array(hi) - np.array(vals)],
            fmt="none",
            ecolor="black",
            capsize=3,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(protocols, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(r"Pr(domain exact) on $\mathbb{F}_{13}^\times$")
    ax.set_title(title)
    ax.legend()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out
