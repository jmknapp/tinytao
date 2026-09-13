#!/usr/bin/env python3
"""Phase 8: substitution test for the dlog embedding story.

Stated falsifier
----------------
Replace first-layer embeddings u, v with their fitted dlog sinusoids
(keeping the original value at residue 0). Leave the hidden bias and
the readout untouched. If complete-domain accuracy collapses, the
dlog fit is correlational: the network was using the residual.

A weaker salvage: same projection, then least-squares refit of W2, b2
onto one-hot labels. If that is exact, the projected hidden layer is
linearly sufficient for the table even if the original readout was not
aligned to the projection.

Control: the same protocol with residue-axis Fourier fits, which have
the same number of free parameters and lost the Phase 5 comparison.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.analysis.ablation import domain_accuracy
from src.analysis.behavior import embeddings
from src.analysis.substitute import (
    project_uv,
    refit_readout_lstsq,
    restore_first_two_layers,
    snapshot_first_two_layers,
    write_embeddings,
)
from src.hardware import resolve_device
from src.population.load import load_successful
from src.repro import environment_record
from src.storage import new_run_dir, write_json
from src.visualization.phase8 import plot_substitute_exact

SOURCES = [
    {
        "name": "width3",
        "width": 3,
        "path": ROOT
        / "results"
        / "runs"
        / "20260909_022909_phase6_minimal"
        / "width3_hunt"
        / "checkpoints"
        / "successful.pt",
    },
    {
        "name": "width4",
        "width": 4,
        "path": ROOT
        / "results"
        / "runs"
        / "20260909_021014_phase3_width_sweep"
        / "width_4"
        / "checkpoints"
        / "successful.pt",
    },
    {
        "name": "width8",
        "width": 8,
        "path": ROOT
        / "results"
        / "runs"
        / "20260909_021014_phase3_width_sweep"
        / "width_8"
        / "checkpoints"
        / "successful.pt",
    },
]

CONDITIONS = [
    {"name": "identity", "family": None, "which": None, "refit": False},
    {"name": "dlog_high", "family": "dlog", "which": "high", "refit": False},
    {"name": "dlog_all", "family": "dlog", "which": "all", "refit": False},
    {"name": "fourier_all", "family": "fourier", "which": "all", "refit": False},
    {"name": "dlog_high_lstsq", "family": "dlog", "which": "high", "refit": True},
    {"name": "dlog_all_lstsq", "family": "dlog", "which": "all", "refit": True},
    {"name": "fourier_all_lstsq", "family": "fourier", "which": "all", "refit": True},
]


def run_condition(model, snap, u, v, spec: dict, p: int, device) -> dict:
    restore_first_two_layers(model, snap)
    info = {"family": spec["family"], "which": spec["which"], "n_replaced": 0, "n_sides": None, "lstsq_resid": None}
    replaced_per_net = None
    if spec["family"] is not None:
        u_p, v_p, info = project_uv(u, v, family=spec["family"], which=spec["which"])
        write_embeddings(model, u_p, v_p)
        replaced_per_net = torch.tensor(info["replaced_per_net"], device=str(device))
    if spec["refit"]:
        info["lstsq_resid"] = refit_readout_lstsq(model, p, device)
    acc = domain_accuracy(model, p, device)
    n = model.population
    exact = acc >= 1.0 - 1e-12
    n_exact = int(exact.sum())
    n_noop = int((replaced_per_net == 0).sum()) if replaced_per_net is not None else n
    if replaced_per_net is not None:
        touched = replaced_per_net > 0
        n_touched = int(touched.sum())
        n_exact_touched = int((exact & touched).sum())
        mean_touched = float(acc[touched].mean()) if n_touched else None
    else:
        n_touched = 0
        n_exact_touched = n_exact
        mean_touched = float(acc.mean())
    return {
        "name": spec["name"],
        "n_exact": n_exact,
        "n": n,
        "frac_exact": n_exact / n,
        "n_noop": n_noop,
        "n_touched": n_touched,
        "n_exact_touched": n_exact_touched,
        "mean_domain": float(acc.mean()),
        "mean_domain_touched": mean_touched,
        "min_domain": float(acc.min()),
        "max_domain": float(acc.max()),
        "n_replaced": info.get("n_replaced"),
        "n_sides": info.get("n_sides"),
        "mean_family_r2": info.get("mean_family_r2"),
        "lstsq_resid": info.get("lstsq_resid"),
        "per_net": acc.detach().cpu().tolist() if n <= 16 else None,
    }


def analyze_source(spec: dict, device, p: int) -> dict:
    model, meta = load_successful(spec["path"], [spec["width"]], 2 * p, p, device=device)
    acc0 = domain_accuracy(model, p, device)
    print(f"{spec['name']}: loaded={model.population} still_exact={int((acc0 >= 1).sum())}")
    u, v, _c = embeddings(model, p)
    snap = snapshot_first_two_layers(model)
    conditions = []
    for cond in CONDITIONS:
        row = run_condition(model, snap, u, v, cond, p, device)
        conditions.append(row)
        extra = ""
        if row.get("n_replaced"):
            extra = f"  replaced {row['n_replaced']}/{row['n_sides']}"
        if row.get("n_noop"):
            extra += f"  noop={row['n_noop']}"
        if row.get("n_touched"):
            extra += f"  exact_touched={row['n_exact_touched']}/{row['n_touched']}"
        if row["lstsq_resid"] is not None:
            extra += f"  lstsq_resid={row['lstsq_resid']:.3e}"
        print(
            f"  {row['name']:18s} exact {row['n_exact']}/{row['n']}  "
            f"mean={row['mean_domain']:.3f} min={row['min_domain']:.3f}{extra}"
        )
    restore_first_two_layers(model, snap)
    return {
        "name": spec["name"],
        "width": spec["width"],
        "n": model.population,
        "indices": meta["indices"].tolist() if torch.is_tensor(meta.get("indices")) else None,
        "conditions": conditions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()
    device = resolve_device("cuda")
    p = 13
    run_dir = new_run_dir(ROOT, "phase8_substitute")
    write_json(run_dir / "meta.json", environment_record(str(ROOT)))
    print(f"run directory: {run_dir}")

    sources = SOURCES
    if args.only:
        sources = [s for s in SOURCES if s["name"] in args.only]
    rows = []
    for spec in sources:
        row = analyze_source(spec, device, p)
        write_json(run_dir / "metrics" / f"{spec['name']}.json", row)
        rows.append(row)
    write_json(run_dir / "metrics" / "substitute.json", {"rows": rows})
    plot_substitute_exact(rows, run_dir / "figures" / "substitute_exact.png")
    print("wrote", run_dir / "metrics" / "substitute.json")


if __name__ == "__main__":
    main()
