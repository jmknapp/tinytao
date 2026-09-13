"""Stage B figures. Generated after training, from master_table.json."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _family_order(rows: list[dict]) -> list[str]:
    order = [
        "genuine",
        "A_isomorphic",
        "B_output_perm",
        "C_slot_relabel",
        "D_random_comm",
        "E_latin_nonassoc",
        "E_latin_comm_nonassoc",
        "F_assoc_damage",
        "G_noncyclic_group",
    ]
    seen = {r["control"] for r in rows}
    return [c for c in order if c in seen]


def _rates(rows: list[dict], key: str) -> tuple[list[str], list[float]]:
    fams = _family_order(rows)
    xs, ys = [], []
    for c in fams:
        sub = [r for r in rows if r["control"] == c]
        n_pop = sum(int(x["population"]) for x in sub)
        n = sum(int(x[key]) for x in sub)
        xs.append(c.replace("_", "\n"))
        ys.append(n / n_pop if n_pop else 0.0)
    return xs, ys


def plot_rate_by_control(rows: list[dict], key: str, ylabel: str, title: str, out: Path) -> None:
    xs, ys = _rates(rows, key)
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    ax.bar(np.arange(len(xs)), ys, color="#3d5a80")
    ax.set_xticks(np.arange(len(xs)))
    ax.set_xticklabels(xs, fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, max(0.05, max(ys) * 1.15 if ys else 1))
    _save(fig, out)


def plot_solver_vs_associativity(rows: list[dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for key, color, label in (
        ("frac_exact", "#3d5a80", "exact"),
        ("frac_faithful", "#2a9d8f", "faithful"),
        ("frac_class2", "#e9c46a", "Class-2"),
    ):
        xs = [r["associativity_fraction"] for r in rows]
        ys = [r[key] for r in rows]
        ax.scatter(xs, ys, s=18, alpha=0.75, c=color, label=label)
    ax.set_xlabel("associativity fraction A")
    ax.set_ylabel("solver fraction")
    ax.set_title("Solver frequency vs associativity fraction")
    ax.legend(frameon=False)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    _save(fig, out)


def plot_train_vs_held(rows: list[dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    colors = {
        "genuine": "#2a9d8f",
        "A_isomorphic": "#2a9d8f",
        "B_output_perm": "#3d5a80",
        "C_slot_relabel": "#e9c46a",
        "D_random_comm": "#e76f51",
        "E_latin_nonassoc": "#e76f51",
        "E_latin_comm_nonassoc": "#e76f51",
        "F_assoc_damage": "#f4a261",
        "G_noncyclic_group": "#264653",
    }
    for r in rows:
        ax.scatter(
            r["mean_train_acc"],
            r["mean_test_acc"],
            s=22,
            alpha=0.8,
            c=colors.get(r["control"], "#888"),
        )
    ax.plot([0, 1], [0, 1], ls="--", c="#bbb", lw=1)
    ax.set_xlabel("mean train accuracy")
    ax.set_ylabel("mean held-row accuracy")
    ax.set_title("Train vs held-row accuracy")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    _save(fig, out)


def plot_dose_response(rows: list[dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    frows = [r for r in rows if r["control"] == "F_assoc_damage"]
    genu = [r for r in rows if r["control"] == "genuine"]
    for p, color in ((11, "#3d5a80"), (19, "#2a9d8f"), (29, "#e76f51")):
        xs, ye, yf, yq = [], [], [], []
        g = [r for r in genu if r["p"] == p]
        if g:
            xs.append(0.0)
            ye.append(g[0]["frac_exact"])
            yf.append(g[0]["frac_faithful"])
            yq.append(g[0]["frac_class2"])
        sub = sorted([r for r in frows if r["p"] == p], key=lambda r: r.get("frac") or 0)
        # mean by dose
        doses = sorted({r["frac"] for r in sub if r.get("frac") is not None})
        for d in doses:
            cell = [r for r in sub if r["frac"] == d]
            n_pop = sum(r["population"] for r in cell)
            xs.append(100.0 * d)
            ye.append(sum(r["n_exact"] for r in cell) / n_pop if n_pop else 0)
            yf.append(sum(r["n_faithful"] for r in cell) / n_pop if n_pop else 0)
            yq.append(sum(r["n_class2"] for r in cell) / n_pop if n_pop else 0)
        ax.plot(xs, ye, marker="o", color=color, label=f"p={p} exact")
        ax.plot(xs, yf, marker="s", ls="--", color=color, alpha=0.7, label=f"p={p} faithful")
        ax.plot(xs, yq, marker="^", ls=":", color=color, alpha=0.7, label=f"p={p} Class-2")
    ax.set_xlabel("associativity damage (% unordered pairs)")
    ax.set_ylabel("solver fraction")
    ax.set_title("Associativity-corruption dose-response")
    ax.legend(frameon=False, fontsize=8, ncol=3)
    ax.set_ylim(-0.02, 1.02)
    _save(fig, out)


def plot_hom_residuals(rows: list[dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    fams = _family_order(rows)
    data = []
    labels = []
    for c in fams:
        vals = [
            r["classification"]["mean_hom_deg"]
            if r.get("classification") and r["classification"].get("mean_hom_deg") is not None
            else r.get("mean_hom_deg")
            for r in rows
            if r["control"] == c
        ]
        vals = [v for v in vals if v is not None]
        if vals:
            data.append(vals)
            labels.append(c.replace("_", "\n"))
    if not data:
        return
    ax.boxplot(data, tick_labels=labels)
    ax.set_ylabel("mean |centered hom. residual| (deg)")
    ax.set_title("Homomorphism residual by control (exact nets)")
    _save(fig, out)


def plot_embedding(xy: np.ndarray, title: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    ax.scatter(xy[:, 0], xy[:, 1], s=28, c="#3d5a80")
    for i in range(xy.shape[0]):
        ax.annotate(str(i), (xy[i, 0], xy[i, 1]), fontsize=7, color="#555")
    ax.set_aspect("equal")
    ax.set_xlabel("Re z(a)")
    ax.set_ylabel("Im z(a)")
    ax.set_title(title)
    ax.axhline(0, c="#ddd", lw=0.8)
    ax.axvline(0, c="#ddd", lw=0.8)
    _save(fig, out)
