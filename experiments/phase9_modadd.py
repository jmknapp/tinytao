#!/usr/bin/env python3
"""Phase 9: modular-addition control for the dlog measurement.

Same architecture, optimizer, seed, and full-domain protocol as the
Phase 3 multiplication sweep. Two questions:

1. Do exact adders win the residue-Fourier family the way multipliers
   won dlog?
2. Does substituting that winning fit preserve exactness, or is
   “high R² plus a load-bearing residual” a generic tiny-ReLU fact?
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from experiments.phase3_width_sweep import run_width
from src.analysis.ablation import domain_accuracy, residue_shift_accuracy
from src.analysis.behavior import embeddings, hidden_maps
from src.analysis.falsify import hidden_factor_r2
from src.analysis.fourier import describe_network_embeddings, family_votes
from src.analysis.substitute import (
    restore_first_two_layers,
    run_substitution_condition,
    snapshot_first_two_layers,
)
from src.config import ExperimentConfig
from src.hardware import resolve_device
from src.population.load import load_successful
from src.repro import environment_record, seed_everything
from src.storage import new_run_dir, save_run_skeleton, write_json
from src.visualization.phase9 import plot_add_substitute, plot_family_compare
from src.visualization.sweep import plot_width_sweep

TASK = "modular_addition"
MUL_SWEEP = ROOT / "results" / "runs" / "20260909_021014_phase3_width_sweep"

SUB_CONDS = [
    {"name": "identity", "family": None, "which": None, "refit": False},
    {"name": "fourier_high", "family": "fourier", "which": "high", "refit": False},
    {"name": "fourier_all", "family": "fourier", "which": "all", "refit": False},
    {"name": "dlog_high", "family": "dlog", "which": "high", "refit": False},
    {"name": "dlog_all", "family": "dlog", "which": "all", "refit": False},
    {"name": "fourier_high_lstsq", "family": "fourier", "which": "high", "refit": True},
]


def _mean_r2(unit_desc: list) -> dict[str, float]:
    f, d, o = [], [], []
    for net in unit_desc:
        for u in net:
            for side in ("u", "v"):
                f.append(u[side]["fourier_r2"])
                d.append(u[side]["dlog_r2"])
                o.append(u[side]["onehot_r2"])
    return {
        "mean_fourier_r2": float(np.mean(f)),
        "mean_dlog_r2": float(np.mean(d)),
        "mean_onehot_r2": float(np.mean(o)),
    }


def analyze_checkpoint(
    path: Path,
    width: int,
    device: torch.device,
    p: int,
    task_name: str,
) -> dict | None:
    if not path.is_file():
        return None
    model, meta = load_successful(path, [width], 2 * p, p, device=device)
    acc = domain_accuracy(model, p, device, task_name=task_name)
    n_exact = int((acc >= 1.0 - 1e-12).sum())
    print(f"  {task_name} width={width} loaded={model.population} still_exact={n_exact}")
    if n_exact == 0:
        return None
    u, v, _c = embeddings(model, p)
    unit_desc = describe_network_embeddings(u, v, p)
    votes = family_votes([u for net in unit_desc for u in net])
    r2 = _mean_r2(unit_desc)
    shift = residue_shift_accuracy(model, p, device, shift=1, task_name=task_name)
    maps = hidden_maps(model, p, device).cpu().numpy()
    factors = [hidden_factor_r2(maps[i, h], p) for i in range(maps.shape[0]) for h in range(maps.shape[1])]
    factor_mean = {k: float(np.mean([f[k] for f in factors])) for k in factors[0]}

    snap = snapshot_first_two_layers(model)
    sub_rows = []
    for spec in SUB_CONDS:
        row = run_substitution_condition(model, snap, u, v, spec, p, device, task_name=task_name)
        sub_rows.append({k: row[k] for k in row if k != "per_net"})
        print(
            f"    {row['name']:20s} exact_touched={row['n_exact_touched']}/{row['n_touched']}  "
            f"mean={row['mean_domain']:.3f}"
        )
    restore_first_two_layers(model, snap)
    return {
        "name": f"width{width}",
        "width": width,
        "n": model.population,
        "n_exact": n_exact,
        "votes": votes,
        "r2": r2,
        "shift_plus1_mean": float(shift.mean()),
        "hidden_factor": factor_mean,
        "substitution": sub_rows,
        "indices": meta["indices"].tolist() if torch.is_tensor(meta.get("indices")) else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "phase9_modadd.yaml")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--sweep-dir", type=Path, default=None)
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    device = resolve_device(cfg.device)
    p = cfg.task.p

    if args.skip_train:
        if args.sweep_dir is None:
            raise SystemExit("--sweep-dir is required with --skip-train")
        run_dir = args.sweep_dir
        print(f"analyzing existing sweep {run_dir}")
        sweep_rows = []
    else:
        seed_everything(cfg.seed)
        run_dir = new_run_dir(ROOT, cfg.run_name)
        save_run_skeleton(run_dir, cfg, environment_record(str(ROOT)))
        print(f"run directory: {run_dir}")
        sweep_rows = []
        for hidden in cfg.sweep.hidden_widths:
            sweep_rows.append(run_width(cfg, int(hidden), device, run_dir))
            write_json(run_dir / "metrics" / "sweep.json", {"rows": sweep_rows})
        plot_width_sweep(sweep_rows, run_dir / "figures")

    add_mech = []
    mul_mech = []
    for hidden in cfg.sweep.hidden_widths:
        add = analyze_checkpoint(
            run_dir / f"width_{hidden}" / "checkpoints" / "successful.pt",
            int(hidden),
            device,
            p,
            TASK,
        )
        if add:
            add_mech.append(add)
            write_json(run_dir / "metrics" / f"add_width_{hidden}.json", add)
        mul_ckpt = MUL_SWEEP / f"width_{hidden}" / "checkpoints" / "successful.pt"
        mul = analyze_checkpoint(mul_ckpt, int(hidden), device, p, "modular_multiplication")
        if mul:
            mul_mech.append(mul)

    write_json(
        run_dir / "metrics" / "compare.json",
        {"addition": add_mech, "multiplication": mul_mech, "sweep": sweep_rows},
    )
    if add_mech and mul_mech:
        # Align on shared width names present in both.
        add_by = {r["name"]: r for r in add_mech}
        mul_by = {r["name"]: r for r in mul_mech}
        shared = [k for k in add_by if k in mul_by]
        if shared:
            plot_family_compare(
                [add_by[k] for k in shared],
                [mul_by[k] for k in shared],
                run_dir / "figures" / "family_compare.png",
            )
    if add_mech:
        plot_add_substitute(add_mech, run_dir / "figures" / "add_substitute.png")
    print("wrote", run_dir / "metrics" / "compare.json")


if __name__ == "__main__":
    main()
