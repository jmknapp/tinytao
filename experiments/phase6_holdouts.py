#!/usr/bin/env python3
"""Phase 6a: structured holdouts at a width that can represent the table."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.stats import wilson_interval
from src.config import ExperimentConfig
from src.hardware import resolve_device
from src.repro import environment_record, seed_everything
from src.runner import finalize_population, fit_and_build, prepare_task, split_kwargs, train_prepared
from src.storage import new_run_dir, write_json
from src.tasks.base import SplitKind
from src.visualization.phase6 import plot_holdouts

HOLDOUTS = [
    {
        "name": "random_70",
        "split": SplitKind.RANDOM,
        "tells": "Ordinary interpolation. A smoothed lookup table can pass. Phase 3 baseline.",
    },
    {
        "name": "held_operand_12",
        "split": SplitKind.HELD_OUT_OPERAND,
        "tells": "Missing a whole row and column of the table. Completing it requires a rule, not local smoothing.",
    },
    {
        "name": "held_zero",
        "split": SplitKind.HELD_OUT_ZERO,
        "tells": "Absorbing element never seen. Tests whether 0 is the same rule or a special case.",
    },
    {
        "name": "held_quadrant",
        "split": SplitKind.HELD_OUT_QUADRANT,
        "tells": "Train on small residues only. Tests extrapolation off a rectangular block.",
    },
    {
        "name": "held_product_0",
        "split": SplitKind.HELD_OUT_PRODUCT,
        "tells": "Never trained to emit residue 0. Tests output-class specialization vs computing the product.",
    },
    {
        "name": "held_dlog_odd",
        "split": SplitKind.HELD_OUT_DLOG_PARITY,
        "tells": "Hold out odd discrete-log rows of a. Tests whether a dlog character interpolates in log space.",
    },
]


def run_one(cfg: ExperimentConfig, spec: dict, device, parent: Path) -> dict:
    local = copy.deepcopy(cfg)
    local.task.split = spec["split"].value
    local.run_name = spec["name"]
    seed_everything(cfg.seed)
    prepared = prepare_task(local, device, protocol="split")
    # prepare_task uses cfg.task.split; dlog/operand kwargs come from split_kwargs.
    if prepared.split.kind != spec["split"]:
        # Force the intended split if protocol mapping differed.
        extra = split_kwargs(local, prepared.task.p)
        prepared.split = prepared.task.split(spec["split"], seed=local.task.split_seed, **extra)
        prepared.train_x = prepared.domain_x[prepared.split.train_idx]
        prepared.train_y = prepared.domain_y[prepared.split.train_idx]
        prepared.test_x = prepared.domain_x[prepared.split.test_idx]
        prepared.test_y = prepared.domain_y[prepared.split.test_idx]

    model, fitted = fit_and_build(local, prepared, device, seed=cfg.seed)
    sub = parent / spec["name"]
    for d in ("metrics", "checkpoints", "figures", "logs"):
        (sub / d).mkdir(parents=True, exist_ok=True)
    print(
        f"{spec['name']}: P={fitted} train={prepared.train_y.numel()} "
        f"test={prepared.test_y.numel()} info={prepared.split.info}"
    )
    result = train_prepared(model, prepared, local, show_progress=True)
    packed = finalize_population(model, result, prepared, sub, write_figures=True)
    counts = packed["counts"]
    p_hat, lo, hi = wilson_interval(counts.exact, counts.total)
    row = {
        "split": spec["name"],
        "kind": spec["split"].value,
        "tells": spec["tells"],
        "info": prepared.split.info,
        "n_train": int(prepared.train_y.numel()),
        "n_test": int(prepared.test_y.numel()),
        "population": fitted,
        "counts": counts.as_dict(),
        "fractions": counts.fractions(),
        "n_exact": counts.exact,
        "frac_exact": p_hat,
        "frac_exact_lo": lo,
        "frac_exact_hi": hi,
        "mean_train_acc": float(result.final_train_acc.mean()),
        "mean_test_acc": float(result.final_test_acc.mean()),
        "mean_domain_acc": float(result.final_domain_acc.mean()),
        "max_domain_acc": float(result.final_domain_acc.max()),
        "max_test_acc": float(result.final_test_acc.max()),
        "seconds": result.seconds,
    }
    write_json(sub / "metrics" / "holdout_row.json", row)
    print(
        f"  exact {counts.exact}/{counts.total} ({p_hat:.3f})  "
        f"train={row['mean_train_acc']:.3f} test={row['mean_test_acc']:.3f} "
        f"domain={row['mean_domain_acc']:.3f} max_domain={row['max_domain_acc']:.3f}"
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "phase6_holdouts.yaml")
    parser.add_argument("--only", type=str, nargs="*", default=None)
    args = parser.parse_args()
    cfg = ExperimentConfig.from_yaml(args.config)
    device = resolve_device(cfg.device)
    run_dir = new_run_dir(ROOT, cfg.run_name)
    write_json(run_dir / "meta.json", environment_record(str(ROOT)))
    print(f"run directory: {run_dir}")

    specs = HOLDOUTS
    if args.only:
        specs = [s for s in HOLDOUTS if s["name"] in args.only]
    rows = []
    for spec in specs:
        rows.append(run_one(cfg, spec, device, run_dir))
        write_json(run_dir / "metrics" / "holdouts.json", {"rows": rows})
    plot_holdouts(rows, run_dir / "figures" / "holdouts.png")
    write_json(run_dir / "metrics" / "holdouts.json", {"rows": rows})
    print("wrote", run_dir / "metrics" / "holdouts.json")


if __name__ == "__main__":
    main()
