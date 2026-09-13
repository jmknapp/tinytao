#!/usr/bin/env python3
"""Class-preserving held row/column on F_13*.

Hold a=12 or b=12. Train keeps all 12 product classes. The held
residue still appears in the other slot, so a shared embedding is
trained and a slot-specific map is not.

Success: domain-exact on all 144 pairs, including the 12 held ones.
"""

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
from src.runner import finalize_population, fit_and_build, prepare_task, train_prepared
from src.storage import new_run_dir, write_json
from src.tasks.base import SplitKind
from src.tasks.modular import ModularMultiplicationStar
from src.visualization.complex import plot_model_exact

MODELS = [
    {"tag": "add", "interaction": "add", "hidden": [4]},
    {"tag": "product", "interaction": "product", "hidden": [4]},
    {"tag": "complex_h1", "interaction": "complex", "hidden": [1]},
    {"tag": "complex_h3", "interaction": "complex", "hidden": [3]},
]

PROTOCOLS = [
    {"name": "held_row", "split": "held_row"},
    {"name": "held_col", "split": "held_col"},
]


def describe_splits() -> None:
    task = ModularMultiplicationStar(p=13)
    print("split sanity (held residue 12):")
    for kind in (SplitKind.HELD_ROW, SplitKind.HELD_COL, SplitKind.DLOG_CHECKERBOARD):
        sp = task.split(kind, seed=0, held_operands=[12])
        a, b, _, y = task.full_domain("cpu")
        tr, te = y[sp.train_idx], y[sp.test_idx]
        print(
            f"  {kind.value}: train={int(sp.train_idx.numel())} test={int(sp.test_idx.numel())} "
            f"train_classes={int(tr.unique().numel())} test_classes={int(te.unique().numel())} "
            f"info={sp.info}"
        )


def run_one(cfg: ExperimentConfig, model_spec: dict, proto: dict, device, parent: Path) -> dict:
    local = copy.deepcopy(cfg)
    local.model.interaction = model_spec["interaction"]
    local.model.hidden_dims = list(model_spec["hidden"])
    local.task.split = proto["split"]
    local.run_name = f"{model_spec['tag']}_{proto['name']}"
    seed_everything(cfg.seed)
    prepared = prepare_task(local, device, protocol="split")
    model, fitted = fit_and_build(local, prepared, device, seed=cfg.seed)
    sub = parent / local.run_name
    for d in ("metrics", "checkpoints", "figures", "logs"):
        (sub / d).mkdir(parents=True, exist_ok=True)
    print(
        f"{model_spec['tag']} {proto['name']}: P={fitted} params={model.n_params_per_network()} "
        f"train={prepared.train_y.numel()} test={prepared.test_y.numel()} "
        f"train_classes={int(prepared.train_y.unique().numel())} "
        f"test_classes={int(prepared.test_y.unique().numel())}"
    )
    result = train_prepared(model, prepared, local, show_progress=True)
    packed = finalize_population(model, result, prepared, sub, write_figures=True)
    counts = packed["counts"]
    p_hat, lo, hi = wilson_interval(counts.exact, counts.total)
    row = {
        "interaction": model_spec["tag"],
        "protocol": proto["name"],
        "split_kind": proto["split"],
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
        "seconds": result.seconds,
        "info": prepared.split.info,
    }
    write_json(sub / "metrics" / "row.json", row)
    print(
        f"  exact {counts.exact}/{counts.total} ({p_hat:.4f})  "
        f"train={row['mean_train_acc']:.3f} test={row['mean_test_acc']:.3f} "
        f"domain={row['mean_domain_acc']:.3f} max_domain={row['max_domain_acc']:.3f} "
        f"max_test={row['max_test_acc']:.3f}"
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "held_row.yaml")
    parser.add_argument("--population", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
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
    describe_splits()

    rows = []
    for model_spec in MODELS:
        for proto in PROTOCOLS:
            rows.append(run_one(cfg, model_spec, proto, device, run_dir))
            write_json(run_dir / "metrics" / "held_row.json", {"rows": rows})
    plot_model_exact(
        rows,
        run_dir / "figures" / "held_row_exact.png",
        title="Held row/col, residue 12, class-preserving",
        model_order=["add", "product", "complex_h1", "complex_h3"],
        model_labels={
            "add": "A add (160)",
            "product": "B product (160)",
            "complex_h1": "C heads=1 (60)",
            "complex_h3": "C heads=3 (156)",
        },
    )
    write_json(run_dir / "metrics" / "held_row.json", {"rows": rows, "n_classes": 12, "chance": 1 / 12})
    print("wrote", run_dir / "metrics" / "held_row.json")


if __name__ == "__main__":
    main()
