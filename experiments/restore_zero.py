#!/usr/bin/env python3
"""Restore zero: Model A vs C on the full 13×13 multiplication table.

F_13* composition is already demonstrated. This asks whether the same
complex-product architecture absorbs 0, and whether a missing factor
(row 12 or row 0) is still filled in.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.analysis.stats import wilson_interval
from src.config import ExperimentConfig
from src.hardware import resolve_device
from src.repro import environment_record, seed_everything
from src.runner import finalize_population, fit_and_build, prepare_task, train_prepared
from src.storage import new_run_dir, write_json
from src.visualization.complex import plot_model_exact

MODELS = [
    {"tag": "add", "interaction": "add", "hidden": [4], "activation": "relu"},
    {"tag": "complex_h1", "interaction": "complex", "hidden": [1], "activation": "identity"},
    {"tag": "complex_h3", "interaction": "complex", "hidden": [3], "activation": "identity"},
]

PROTOCOLS = [
    {"name": "full_table", "protocol": "full_domain", "split": "random", "held": [12]},
    {"name": "random_70", "protocol": "split", "split": "random", "held": [12]},
    {"name": "held_row_12", "protocol": "split", "split": "held_row", "held": [12]},
    {"name": "held_row_0", "protocol": "split", "split": "held_row", "held": [0]},
]


@torch.no_grad()
def zero_radius_ratio(model, exact_mask: torch.Tensor) -> dict | None:
    if not hasattr(model, "E") or not bool(exact_mask.any()):
        return None
    # E: [P, n, H, 2]; residue 0 is index 0
    r0 = model.E[:, 0].norm(dim=-1).mean(dim=1)
    r_nz = model.E[:, 1:].norm(dim=-1).mean(dim=(1, 2))
    r0_e = r0[exact_mask]
    r_nz_e = r_nz[exact_mask]
    ratio = r0_e / (r_nz_e + 1e-12)
    return {
        "n_exact": int(exact_mask.sum()),
        "mean_r0": float(r0_e.mean()),
        "mean_r_nonzero": float(r_nz_e.mean()),
        "mean_r0_over_r_nz": float(ratio.mean()),
        "median_r0_over_r_nz": float(ratio.median()),
    }


def run_one(cfg: ExperimentConfig, model_spec: dict, proto: dict, device, parent: Path) -> dict:
    local = copy.deepcopy(cfg)
    local.model.interaction = model_spec["interaction"]
    local.model.hidden_dims = list(model_spec["hidden"])
    local.model.activation = model_spec["activation"]
    local.task.split = proto["split"]
    local.task.held_operands = list(proto["held"])
    local.run_name = f"{model_spec['tag']}_{proto['name']}"
    seed_everything(cfg.seed)
    prepared = prepare_task(local, device, protocol=proto["protocol"])
    model, fitted = fit_and_build(local, prepared, device, seed=cfg.seed)
    sub = parent / local.run_name
    for d in ("metrics", "checkpoints", "figures", "logs"):
        (sub / d).mkdir(parents=True, exist_ok=True)
    print(
        f"{model_spec['tag']} {proto['name']}: P={fitted} params={model.n_params_per_network()} "
        f"train={prepared.train_y.numel()} test={prepared.test_y.numel()} "
        f"train_classes={int(prepared.train_y.unique().numel())} "
        f"test_classes={int(prepared.test_y.unique().numel())} "
        f"domain={prepared.task.n_domain}"
    )
    result = train_prepared(model, prepared, local, show_progress=True)
    packed = finalize_population(model, result, prepared, sub, write_figures=True)
    counts = packed["counts"]
    p_hat, lo, hi = wilson_interval(counts.exact, counts.total)
    exact_mask = packed["labels"] == 4
    radii = zero_radius_ratio(model, exact_mask)
    row = {
        "interaction": model_spec["tag"],
        "protocol": proto["name"],
        "split_kind": proto["split"],
        "held_operands": list(proto["held"]),
        "n_train": int(prepared.train_y.numel()),
        "n_test": int(prepared.test_y.numel()),
        "n_train_classes": int(prepared.train_y.unique().numel()),
        "n_test_classes": int(prepared.test_y.unique().numel()),
        "n_domain": prepared.task.n_domain,
        "params": model.n_params_per_network(),
        "population": fitted,
        "counts": counts.as_dict(),
        "n_exact": counts.exact,
        "frac_exact": p_hat,
        "frac_exact_lo": lo,
        "frac_exact_hi": hi,
        "mean_train_acc": float(result.final_train_acc.mean()),
        "mean_test_acc": float(result.final_test_acc.mean()),
        "mean_domain_acc": float(result.final_domain_acc.mean()),
        "max_domain_acc": float(result.final_domain_acc.max()),
        "max_test_acc": float(result.final_test_acc.max()),
        "zero_radius": radii,
        "seconds": result.seconds,
        "info": prepared.split.info,
    }
    write_json(sub / "metrics" / "row.json", row)
    extra = ""
    if radii:
        extra = f"  r0/r*={radii['mean_r0_over_r_nz']:.3f}"
    print(
        f"  exact {counts.exact}/{counts.total} ({p_hat:.4f})  "
        f"train={row['mean_train_acc']:.3f} test={row['mean_test_acc']:.3f} "
        f"domain={row['mean_domain_acc']:.3f} max_domain={row['max_domain_acc']:.3f}"
        f"{extra}"
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "restore_zero.yaml")
    parser.add_argument("--population", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--models", nargs="+", default=None, help="subset of tags: add complex_h1 complex_h3")
    args = parser.parse_args()
    cfg = ExperimentConfig.from_yaml(args.config)
    if args.population:
        cfg.population.size = args.population
    if args.epochs:
        cfg.train.epochs = args.epochs
    device = resolve_device(cfg.device)
    run_dir = new_run_dir(ROOT, cfg.run_name)
    write_json(run_dir / "meta.json", environment_record(str(ROOT)))
    print(f"run directory: {run_dir}")

    models = MODELS
    if args.models:
        wanted = set(args.models)
        models = [m for m in MODELS if m["tag"] in wanted]
        if not models:
            raise SystemExit(f"no models matched {sorted(wanted)}")
    rows = []
    for model_spec in models:
        for proto in PROTOCOLS:
            rows.append(run_one(cfg, model_spec, proto, device, run_dir))
            write_json(run_dir / "metrics" / "restore_zero.json", {"rows": rows})
    plot_model_exact(
        rows,
        run_dir / "figures" / "restore_zero_exact.png",
        title="Restore zero: A vs C on Z/13Z multiply",
        model_order=["add", "complex_h1", "complex_h3"],
        model_labels={
            "add": "A add (173)",
            "complex_h1": "C heads=1 (65)",
            "complex_h3": "C heads=3 (169)",
        },
    )
    write_json(
        run_dir / "metrics" / "restore_zero.json",
        {"rows": rows, "n_classes": 13, "chance": 1 / 13, "n_domain": 169},
    )
    print("wrote", run_dir / "metrics" / "restore_zero.json")


if __name__ == "__main__":
    main()
