#!/usr/bin/env python3
"""Throughput / VRAM probe for the vectorized population."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.config import ExperimentConfig
from src.hardware import fit_population_size, resolve_device
from src.population.mlp import PopulationMLP
from src.repro import seed_everything
from src.tasks.modular import build_task
from src.training.loop import train_population


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "phase1_modmul.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--population", type=int, default=0, help="0 = use config / fit")
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    device = resolve_device(cfg.device)
    seed_everything(cfg.seed)
    task = build_task(cfg.task.name, cfg.task.p)
    _, _, x, y = task.full_domain(device)

    def factory(p_size: int) -> PopulationMLP:
        return PopulationMLP(
            population=p_size,
            input_dim=task.input_dim,
            hidden_dims=cfg.model.hidden_dims,
            output_dim=task.output_dim,
            activation=cfg.model.activation,
            device=device,
        )

    requested = args.population or cfg.population.size
    p = fit_population_size(factory, x, y, start=requested, minimum=cfg.population.min_size)
    seed_everything(cfg.seed)
    model = factory(p)
    result = train_population(
        model,
        x,
        y,
        x[:1],
        y[:1],
        x,
        y,
        epochs=args.epochs,
        lr=cfg.train.lr,
        optimizer_name=cfg.train.optimizer,
        weight_decay=0.0,
        batch_size=0,
        log_every=max(args.epochs, 1),
        eval_every=max(args.epochs, 1),
        show_progress=True,
    )
    print(f"device                 {device}")
    print(f"population             {p}")
    print(f"domain size            {task.n_domain}")
    print(f"params / network       {model.n_params_per_network()}")
    print(f"step_ms                {result.step_ms:.3f}")
    print(f"examples/sec           {result.examples_per_sec:,.0f}")
    print(f"network-evals/sec      {result.network_evals_per_sec:,.0f}")
    print(f"peak_vram_bytes        {result.peak_vram_bytes}")
    print(f"peak_vram_gib          {result.peak_vram_bytes / (1024**3):.3f}")
    if device.type == "cuda":
        print(f"gpu                    {torch.cuda.get_device_name(device)}")


if __name__ == "__main__":
    main()
