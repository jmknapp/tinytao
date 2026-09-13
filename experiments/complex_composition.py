#!/usr/bin/env python3
"""Model C (complex composition) on F_13*.

Shared 2-D embeddings, interaction z(a) z(b), linear readout.
No discrete-log initialization. Success is domain-exact on all 144
pairs, including those withheld by the split.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.analysis.fourier import discrete_log_table
from src.analysis.stats import wilson_interval
from src.config import ExperimentConfig
from src.hardware import resolve_device
from src.population.complex import PopulationComplexMLP, sequential_complex_logits
from src.repro import environment_record, seed_everything
from src.runner import finalize_population, fit_and_build, prepare_task, train_prepared
from src.storage import new_run_dir, write_json
from src.tasks.modular import ModularMultiplicationStar
from src.visualization.complex import plot_model_exact

PROTOCOLS = [
    {"name": "full_table", "protocol": "full_domain", "split": "random"},
    {"name": "random_70", "protocol": "split", "split": "random"},
    {"name": "dlog_checkerboard", "protocol": "split", "split": "dlog_checkerboard"},
]


def check_complex_forward(device: torch.device) -> None:
    task = ModularMultiplicationStar(p=13)
    _, _, x, _ = task.full_domain(device)
    model = PopulationComplexMLP(8, task.input_dim, [2], task.output_dim, device=device)
    batched = model(x)
    seq = torch.stack([sequential_complex_logits(model, x, i) for i in range(8)], dim=1)
    err = (batched - seq).abs().max().item()
    print(f"complex batched vs sequential max abs {err:.3e}")
    if err > 1e-5:
        raise RuntimeError("complex forward does not match sequential reference")


@torch.no_grad()
def planted_roots_accuracy(device: torch.device, heads: int) -> float:
    """Plant e(a)=exp(2πi dlog(a)/12); nearest-root readout on head 0."""
    task = ModularMultiplicationStar(p=13)
    _, _, x, y = task.full_domain(device)
    model = PopulationComplexMLP(1, task.input_dim, [heads], task.output_dim, device=device)
    dlog = discrete_log_table(task.p)
    n = task.n_group
    W = torch.zeros(2 * heads, task.output_dim, device=device, dtype=model.W_out.dtype)
    for i in range(n):
        ang = 2.0 * math.pi * int(dlog[i + 1]) / n
        model.E[0, i, :, 0] = math.cos(ang)
        model.E[0, i, :, 1] = math.sin(ang)
        W[0, i] = math.cos(ang)
        W[1, i] = math.sin(ang)
    model.W_out[0] = W
    model.b_out[0] = 0
    acc = float((model(x)[:, 0].argmax(dim=-1) == y).float().mean())
    print(f"planted dlog embeddings + nearest-root readout: heads={heads} domain acc={acc:.4f}")
    return acc


def run_one(cfg: ExperimentConfig, heads: int, spec: dict, device, parent: Path) -> dict:
    local = copy.deepcopy(cfg)
    local.model.interaction = "complex"
    local.model.hidden_dims = [heads]
    local.task.split = spec["split"]
    local.run_name = f"complex_h{heads}_{spec['name']}"
    seed_everything(cfg.seed)
    prepared = prepare_task(local, device, protocol=spec["protocol"])
    model, fitted = fit_and_build(local, prepared, device, seed=cfg.seed)
    sub = parent / local.run_name
    for d in ("metrics", "checkpoints", "figures", "logs"):
        (sub / d).mkdir(parents=True, exist_ok=True)
    print(
        f"complex H={heads} {spec['name']}: P={fitted} params={model.n_params_per_network()} "
        f"train={prepared.train_y.numel()} test={prepared.test_y.numel()} "
        f"domain={prepared.task.n_domain}"
    )
    result = train_prepared(model, prepared, local, show_progress=True)
    packed = finalize_population(model, result, prepared, sub, write_figures=True)
    counts = packed["counts"]
    p_hat, lo, hi = wilson_interval(counts.exact, counts.total)
    row = {
        "interaction": f"complex_h{heads}",
        "heads": heads,
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
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "complex.yaml")
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
    check_complex_forward(device)
    planted = {h: planted_roots_accuracy(device, h) for h in (1, 3)}
    if min(planted.values()) < 1.0 - 1e-6:
        raise RuntimeError("planted roots-of-unity + nearest-root readout is not exact; architecture cannot express the group law")

    rows = []
    for heads in (1, 3):
        for spec in PROTOCOLS:
            rows.append(run_one(cfg, heads, spec, device, run_dir))
            write_json(run_dir / "metrics" / "complex.json", {"rows": rows, "planted_nearest_root": planted})
    plot_model_exact(
        rows,
        run_dir / "figures" / "complex_exact.png",
        title="Model C complex composition",
        model_order=["complex_h1", "complex_h3"],
        model_labels={"complex_h1": "C heads=1 (60 params)", "complex_h3": "C heads=3 (156 params)"},
    )
    ab_path = ROOT / "results" / "runs" / "20260909_103432_ab_product" / "metrics" / "ab.json"
    if ab_path.exists():
        ab_rows = json.loads(ab_path.read_text())["rows"]
        combined = ab_rows + rows
        plot_model_exact(
            combined,
            run_dir / "figures" / "abc_exact.png",
            title="A vs B vs C on F_13*",
            model_order=["add", "product", "complex_h1", "complex_h3"],
            model_labels={
                "add": "A add (160)",
                "product": "B product (160)",
                "complex_h1": "C heads=1 (60)",
                "complex_h3": "C heads=3 (156)",
            },
        )
    write_json(
        run_dir / "metrics" / "complex.json",
        {"rows": rows, "planted_nearest_root": planted, "n_classes": 12, "chance": 1 / 12},
    )
    print("wrote", run_dir / "metrics" / "complex.json")


if __name__ == "__main__":
    main()
