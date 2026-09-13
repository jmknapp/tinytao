#!/usr/bin/env python3
"""Phase 1: train thousands of tiny networks as one GPU batch.

Proves the vectorized population implementation, writes a reproducible run
directory, and records whether any network becomes an exact solver of the
fully enumerated domain. Interpretations are out of scope.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.analysis.outcomes import select_representatives, summarize_population
from src.config import ExperimentConfig
from src.correctness import run_all
from src.hardware import fit_population_size, resolve_device
from src.population.mlp import PopulationMLP
from src.repro import environment_record, seed_everything
from src.storage import new_run_dir, save_checkpoints, save_metrics, save_run_skeleton, write_json
from src.tasks.base import SplitKind
from src.tasks.modular import build_task
from src.training.loop import train_population
from src.training.metrics import classify_outcomes, detect_transitions
from src.visualization.plots import plot_phase1


def _split_kwargs(cfg: ExperimentConfig, p: int) -> dict:
    held_ops = [p - 1 if v < 0 else v for v in cfg.task.held_operands]
    kwargs: dict = {
        "train_frac": cfg.task.train_frac,
        "held_operands": held_ops,
        "held_products": list(cfg.task.held_products),
    }
    if cfg.task.cut is not None:
        kwargs["cut"] = cfg.task.cut
    return kwargs


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 vectorized population training")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "phase1_modmul.yaml")
    parser.add_argument("--smoke", action="store_true", help="tiny run for plumbing checks")
    parser.add_argument("--skip-correctness", action="store_true")
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    if args.smoke:
        cfg.population.size = min(cfg.population.size, 128)
        cfg.train.epochs = min(cfg.train.epochs, 40)
        cfg.train.log_every = 5
        cfg.train.eval_every = 5
        cfg.run_name = f"{cfg.run_name}_smoke"

    device = resolve_device(cfg.device)
    seed_everything(cfg.seed)

    run_dir = new_run_dir(ROOT, cfg.run_name)
    print(f"run directory: {run_dir}")
    print(f"device: {device}")

    if not args.skip_correctness:
        print("running correctness checks...")
        checks = run_all(device)
        write_json(run_dir / "logs" / "correctness.json", checks)
        print("  forward max abs error:", checks["forward"]["max_abs_forward"])
        print("  gradient leak:", checks["grad_isolation"]["max_abs_leak"])

    task = build_task(cfg.task.name, cfg.task.p)
    a, b, domain_x, domain_y = task.full_domain(device)
    split = task.split(SplitKind(cfg.task.split), seed=cfg.task.split_seed, **_split_kwargs(cfg, task.p))
    train_x, train_y = domain_x[split.train_idx], domain_y[split.train_idx]
    test_x, test_y = domain_x[split.test_idx], domain_y[split.test_idx]

    def factory(p_size: int) -> PopulationMLP:
        return PopulationMLP(
            population=p_size,
            input_dim=task.input_dim,
            hidden_dims=cfg.model.hidden_dims,
            output_dim=task.output_dim,
            activation=cfg.model.activation,
            use_bias=cfg.model.use_bias,
            init=cfg.model.init,
            init_scale=cfg.model.init_scale,
            device=device,
        )

    print(f"sizing population (requested {cfg.population.size})...")
    fitted_p = fit_population_size(
        factory,
        train_x,
        train_y,
        start=cfg.population.size,
        minimum=cfg.population.min_size,
    )
    if fitted_p != cfg.population.size:
        print(f"  reduced population {cfg.population.size} -> {fitted_p} (VRAM)")
    seed_everything(cfg.seed)
    model = factory(fitted_p)

    meta = environment_record(str(ROOT))
    meta.update(
        {
            "fitted_population": fitted_p,
            "requested_population": cfg.population.size,
            "task": task.describe(),
            "model": model.describe(),
            "split": {
                "kind": split.kind.value,
                "info": split.info,
                "n_train": int(split.train_idx.numel()),
                "n_test": int(split.test_idx.numel()),
                "n_domain": task.n_domain,
            },
        }
    )
    save_run_skeleton(run_dir, cfg, meta)

    print(
        f"training P={fitted_p} nets, hidden={cfg.model.hidden_dims}, "
        f"{cfg.train.epochs} epochs, train={int(split.train_idx.numel())}/{task.n_domain}"
    )
    result = train_population(
        model,
        train_x,
        train_y,
        test_x,
        test_y,
        domain_x,
        domain_y,
        epochs=cfg.train.epochs,
        lr=cfg.train.lr,
        optimizer_name=cfg.train.optimizer,
        weight_decay=cfg.train.weight_decay,
        batch_size=cfg.train.batch_size,
        log_every=cfg.train.log_every,
        eval_every=cfg.train.eval_every,
        grad_clip=cfg.train.grad_clip,
    )

    labels, counts = classify_outcomes(
        result.final_train_acc,
        result.final_test_acc,
        result.final_domain_acc,
        n_classes=task.output_dim,
    )
    sudden = detect_transitions(result.domain_acc)["sudden_transition"]
    summary = summarize_population(
        labels,
        counts,
        result.final_train_acc,
        result.final_test_acc,
        result.final_domain_acc,
        model.param_l2().detach().cpu(),
        result.domain_acc,
        n_classes=task.output_dim,
    )
    summary["throughput"] = {
        "seconds": result.seconds,
        "step_ms": result.step_ms,
        "examples_per_sec": result.examples_per_sec,
        "network_evals_per_sec": result.network_evals_per_sec,
        "peak_vram_bytes": result.peak_vram_bytes,
        "peak_vram_gib": result.peak_vram_bytes / (1024**3),
        "population": fitted_p,
        "epochs": result.epochs_run,
    }
    write_json(run_dir / "metrics" / "summary.json", summary)

    exact_idx = torch.where(labels == 4)[0]
    successful = None
    if exact_idx.numel():
        cap = exact_idx[:128]
        successful = model.slice_state_dict(cap)
        successful["domain_acc"] = result.final_domain_acc[cap]
        successful["labels"] = labels[cap]
    representatives = select_representatives(model, labels, result.final_domain_acc, sudden)

    save_metrics(
        run_dir,
        {
            "labels": labels,
            "train_acc": result.final_train_acc,
            "test_acc": result.final_test_acc,
            "domain_acc": result.final_domain_acc,
            "loss": result.final_loss,
            "log_epochs": torch.tensor(result.log_epochs),
            "sudden_transition": sudden,
        },
        {
            "loss": result.loss,
            "train_acc": result.train_acc,
            "test_acc": result.test_acc,
            "domain_acc": result.domain_acc,
            "param_norm": result.param_norm,
            "grad_norm": result.grad_norm,
            "hidden_mean": result.hidden_mean,
            "hidden_sparsity": result.hidden_sparsity,
            "log_epochs": torch.tensor(result.log_epochs),
        },
    )
    save_checkpoints(run_dir, successful, representatives)
    figures = plot_phase1(
        run_dir / "figures",
        result.log_epochs,
        result.train_acc,
        result.test_acc,
        result.domain_acc,
        result.final_domain_acc,
        labels,
        n_classes=task.output_dim,
    )

    print("outcome counts:", counts.as_dict())
    print("fractions:", {k: f"{v:.3f}" for k, v in counts.fractions().items()})
    print(f"time: {result.seconds:.2f}s  step: {result.step_ms:.2f}ms")
    print(f"examples/s: {result.examples_per_sec:,.0f}  net-evals/s: {result.network_evals_per_sec:,.0f}")
    if result.peak_vram_bytes:
        print(f"peak VRAM: {result.peak_vram_bytes / (1024**3):.2f} GiB")
    print("figures:")
    for p in figures:
        print(" ", p)
    print("summary:", run_dir / "metrics" / "summary.json")


if __name__ == "__main__":
    main()
