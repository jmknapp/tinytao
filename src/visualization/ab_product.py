"""A vs B comparison figure."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_ab_exact(rows: list[dict[str, Any]], out: Path) -> Path:
    protocols = []
    for r in rows:
        if r["protocol"] not in protocols:
            protocols.append(r["protocol"])
    models = ["add", "product"]
    x = np.arange(len(protocols))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    colors = {"add": "#293241", "product": "#ee6c4d"}
    labels = {"add": "A additive-ReLU", "product": "B product"}
    for i, m in enumerate(models):
        vals = []
        lo = []
        hi = []
        for proto in protocols:
            hit = next(r for r in rows if r["protocol"] == proto and r["interaction"] == m)
            vals.append(hit["frac_exact"])
            lo.append(hit["frac_exact_lo"])
            hi.append(hit["frac_exact_hi"])
        offset = (i - 0.5) * width
        ax.bar(x + offset, vals, width, color=colors[m], label=labels[m])
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
    ax.set_title("Model A vs Model B, hidden 4, 160 params")
    ax.legend()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out
