#!/usr/bin/env python3
"""Model A (additive ReLU) vs Model B (product) on F_13*.

Primary success: domain-exact on the 144 nonzero pairs, including
pairs withheld by the split. Full-table exactness is reported but
is not the claim.
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
from src.population.product import PopulationProductMLP, sequential_product_logits
from src.repro import environment_record, seed_everything
from src.runner import finalize_population, fit_and_build, prepare_task, train_prepared
from src.storage import new_run_dir, write_json
from src.tasks.modular import ModularMultiplicationStar
from src.visualization.ab_product import plot_ab_exact

PROTOCOLS = [
    {"name": "full_table", "protocol": "full_domain", "split": "random"},
    {"name": "random_70", "protocol": "split", "split": "random"},
    {"name": "dlog_checkerboard", "protocol": "split", "split": "dlog_checkerboard"},
]


def check_product_forward(device: torch.device) -> None:
    task = ModularMultiplicationStar(p=13)
    _, _, x, _ = task.full_domain(device)
    model = PopulationProductMLP(8, task.input_dim, [4], task.output_dim, device=device)
    batched = model(x)
    seq = torch.stack([sequential_product_logits(model, x, i) for i in range(8)], dim=1)
    err = (batched - seq).abs().max().item()
    print(f"product batched vs sequential max abs {err:.3e}")
    if err > 1e-5:
        raise RuntimeError("product forward does not match sequential reference")


def run_one(cfg: ExperimentConfig, interaction: str, spec: dict, device, parent: Path) -> dict:
    local = copy.deepcopy(cfg)
    local.model.interaction = interaction
    local.task.split = spec["split"]
    local.run_name = f"{interaction}_{spec['name']}"
    seed_everything(cfg.seed)
    prepared = prepare_task(local, device, protocol=spec["protocol"])
    model, fitted = fit_and_build(local, prepared, device, seed=cfg.seed)
    sub = parent / local.run_name
    for d in ("metrics", "checkpoints", "figures", "logs"):
        (sub / d).mkdir(parents=True, exist_ok=True)
    print(
        f"{interaction} {spec['name']}: P={fitted} params={model.n_params_per_network()} "
        f"train={prepared.train_y.numel()} test={prepared.test_y.numel()} "
        f"domain={prepared.task.n_domain}"
    )
    result = train_prepared(model, prepared, local, show_progress=True)
    packed = finalize_population(model, result, prepared, sub, write_figures=True)
    counts = packed["counts"]
    p_hat, lo, hi = wilson_interval(counts.exact, counts.total)
    # Exact generalizer: domain exact AND (for splits) test exact, which
    # is implied by domain exact on an enumerated domain.
    row = {
        "interaction": interaction,
        "protocol": spec["name"],
        "split_kind": spec["split"],
        "n_train": int(prepared.train_y.numel()),
        "n_test": int(prepared.test_y.numel()),
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
        f"domain={row['mean_domain_acc']:.3f} max_domain={row['max_domain_acc']:.3f}"
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "ab_product.yaml")
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
    check_product_forward(device)

    rows = []
    for interaction in ("add", "product"):
        for spec in PROTOCOLS:
            rows.append(run_one(cfg, interaction, spec, device, run_dir))
            write_json(run_dir / "metrics" / "ab.json", {"rows": rows})
    plot_ab_exact(rows, run_dir / "figures" / "ab_exact.png")
    write_json(run_dir / "metrics" / "ab.json", {"rows": rows, "n_classes": 12, "chance": 1 / 12})
    print("wrote", run_dir / "metrics" / "ab.json")


if __name__ == "__main__":
    main()
