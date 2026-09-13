"""Shared single-population experiment used by Phase 1 and sweeps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.analysis.outcomes import select_representatives, summarize_population
from src.config import ExperimentConfig
from src.hardware import fit_population_size
from src.population.complex import PopulationComplexMLP
from src.population.kirchhoff import PopulationKirchhoff
from src.population.mlp import PopulationMLP
from src.population.product import PopulationProductMLP
from src.population.tropical import PopulationTropicalMLP
from src.repro import seed_everything
from src.storage import save_checkpoints, save_metrics, write_json
from src.tasks.base import SplitKind, TaskSplit
from src.tasks.modular import build_task
from src.training.loop import TrainResult, train_population
from src.training.metrics import classify_outcomes, detect_transitions
from src.visualization.plots import plot_phase1


@dataclass
class PreparedTask:
    task: Any
    domain_x: torch.Tensor
    domain_y: torch.Tensor
    train_x: torch.Tensor
    train_y: torch.Tensor
    test_x: torch.Tensor
    test_y: torch.Tensor
    split: TaskSplit
    protocol: str


def split_kwargs(cfg: ExperimentConfig, p: int) -> dict:
    held_ops = [p - 1 if v < 0 else v for v in cfg.task.held_operands]
    kwargs: dict = {
        "train_frac": cfg.task.train_frac,
        "held_operands": held_ops,
        "held_products": list(cfg.task.held_products),
    }
    if cfg.task.cut is not None:
        kwargs["cut"] = cfg.task.cut
    return kwargs


def prepare_from_task(task: Any, cfg: ExperimentConfig, device: torch.device, protocol: str | None = None) -> PreparedTask:
    _, _, domain_x, domain_y = task.full_domain(device)
    proto = protocol or getattr(cfg, "protocol", None) or "split"
    if proto in {"full_domain", "full"}:
        n = domain_y.shape[0]
        idx = torch.arange(n)
        split = TaskSplit(train_idx=idx, test_idx=idx, kind=SplitKind.RANDOM, info={"protocol": "full_domain"})
    else:
        split = task.split(SplitKind(cfg.task.split), seed=cfg.task.split_seed, **split_kwargs(cfg, task.p))
    return PreparedTask(
        task=task,
        domain_x=domain_x,
        domain_y=domain_y,
        train_x=domain_x[split.train_idx],
        train_y=domain_y[split.train_idx],
        test_x=domain_x[split.test_idx],
        test_y=domain_y[split.test_idx],
        split=split,
        protocol=proto,
    )


def prepare_task(cfg: ExperimentConfig, device: torch.device, protocol: str | None = None) -> PreparedTask:
    task = build_task(cfg.task.name, cfg.task.p)
    return prepare_from_task(task, cfg, device, protocol=protocol)


def make_model(cfg: ExperimentConfig, task, population: int, device: torch.device):
    kind = getattr(cfg.model, "interaction", "add")
    kwargs = dict(
        population=population,
        input_dim=task.input_dim,
        hidden_dims=cfg.model.hidden_dims,
        output_dim=task.output_dim,
        activation=cfg.model.activation,
        use_bias=cfg.model.use_bias,
        init=cfg.model.init,
        init_scale=cfg.model.init_scale,
        device=device,
    )
    if kind == "product":
        return PopulationProductMLP(**kwargs)
    if kind == "complex":
        return PopulationComplexMLP(**kwargs)
    if kind == "tropical":
        return PopulationTropicalMLP(**kwargs)
    if kind == "kirchhoff":
        return PopulationKirchhoff(
            **kwargs,
            n_vertices=task.n,
            elist=task.elist,
        )
    if kind == "add":
        return PopulationMLP(**kwargs)
    raise ValueError(f"unknown interaction {kind!r}")


def fit_and_build(
    cfg: ExperimentConfig,
    prepared: PreparedTask,
    device: torch.device,
    seed: int,
) -> tuple[PopulationMLP, int]:
    def factory(p_size: int) -> PopulationMLP:
        return make_model(cfg, prepared.task, p_size, device)

    fitted = fit_population_size(
        factory,
        prepared.train_x,
        prepared.train_y,
        start=cfg.population.size,
        minimum=cfg.population.min_size,
    )
    seed_everything(seed)
    return factory(fitted), fitted


def train_prepared(
    model: PopulationMLP,
    prepared: PreparedTask,
    cfg: ExperimentConfig,
    show_progress: bool = True,
    on_log_step=None,
) -> TrainResult:
    return train_population(
        model,
        prepared.train_x,
        prepared.train_y,
        prepared.test_x,
        prepared.test_y,
        prepared.domain_x,
        prepared.domain_y,
        epochs=cfg.train.epochs,
        lr=cfg.train.lr,
        optimizer_name=cfg.train.optimizer,
        weight_decay=cfg.train.weight_decay,
        batch_size=cfg.train.batch_size,
        log_every=cfg.train.log_every,
        eval_every=cfg.train.eval_every,
        grad_clip=cfg.train.grad_clip,
        show_progress=show_progress,
        on_log_step=on_log_step,
    )


def finalize_population(
    model: PopulationMLP,
    result: TrainResult,
    prepared: PreparedTask,
    run_dir: Path | None,
    write_figures: bool = True,
    max_exact_saved: int = 256,
) -> dict[str, Any]:
    labels, counts = classify_outcomes(
        result.final_train_acc,
        result.final_test_acc,
        result.final_domain_acc,
        n_classes=prepared.task.output_dim,
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
        n_classes=prepared.task.output_dim,
    )
    summary["throughput"] = {
        "seconds": result.seconds,
        "step_ms": result.step_ms,
        "examples_per_sec": result.examples_per_sec,
        "network_evals_per_sec": result.network_evals_per_sec,
        "peak_vram_bytes": result.peak_vram_bytes,
        "peak_vram_gib": result.peak_vram_bytes / (1024**3),
        "population": model.population,
        "epochs": result.epochs_run,
    }
    summary["architecture"] = model.describe()
    summary["protocol"] = prepared.protocol
    summary["n_train"] = int(prepared.train_y.numel())
    summary["n_test"] = int(prepared.test_y.numel())
    summary["n_domain"] = prepared.task.n_domain

    if run_dir is not None:
        write_json(run_dir / "metrics" / "summary.json", summary)
        exact_idx = torch.where(labels == 4)[0]
        successful = None
        if exact_idx.numel():
            cap = exact_idx[:max_exact_saved]
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
        if write_figures:
            plot_phase1(
                run_dir / "figures",
                result.log_epochs,
                result.train_acc,
                result.test_acc,
                result.domain_acc,
                result.final_domain_acc,
                labels,
                n_classes=prepared.task.output_dim,
            )

    return {
        "labels": labels,
        "counts": counts,
        "summary": summary,
        "sudden": sudden,
        "result": result,
    }
