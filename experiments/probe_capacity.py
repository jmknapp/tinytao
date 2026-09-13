#!/usr/bin/env python3
"""Quick probe: can a wider net become an exact solver of the full table?"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.config import ExperimentConfig, ModelConfig, PopulationConfig, TaskConfig, TrainConfig
from src.hardware import resolve_device
from src.repro import seed_everything
from src.runner import fit_and_build, prepare_task, train_prepared


def run_probe(hidden: int, epochs: int, lr: float, optimizer: str, wd: float, p_size: int) -> dict:
    cfg = ExperimentConfig(
        task=TaskConfig(name="modular_multiplication", p=13),
        model=ModelConfig(hidden_dims=[hidden]),
        population=PopulationConfig(size=p_size, min_size=32),
        train=TrainConfig(
            epochs=epochs,
            lr=lr,
            optimizer=optimizer,
            weight_decay=wd,
            log_every=max(epochs // 10, 1),
            eval_every=max(epochs // 10, 1),
        ),
        seed=0,
    )
    device = resolve_device("cuda")
    seed_everything(0)
    prepared = prepare_task(cfg, device, protocol="full_domain")
    model, fitted = fit_and_build(cfg, prepared, device, seed=0)
    result = train_prepared(model, prepared, cfg, show_progress=True)
    exact = int((result.final_domain_acc >= 1.0).sum())
    return {
        "hidden": hidden,
        "params": model.n_params_per_network(),
        "P": fitted,
        "epochs": epochs,
        "lr": lr,
        "opt": optimizer,
        "wd": wd,
        "exact": exact,
        "frac_exact": exact / fitted,
        "mean_domain": float(result.final_domain_acc.mean()),
        "max_domain": float(result.final_domain_acc.max()),
        "mean_train": float(result.final_train_acc.mean()),
        "seconds": result.seconds,
    }


def main() -> None:
    probes = [
        (16, 4000, 3e-3, "adam", 0.0, 256),
        (32, 4000, 3e-3, "adam", 0.0, 256),
        (32, 4000, 1e-3, "adamw", 0.1, 256),
        (24, 8000, 5e-3, "adam", 0.0, 256),
    ]
    rows = []
    for hidden, epochs, lr, opt, wd, p_size in probes:
        print(f"\n=== hidden={hidden} epochs={epochs} lr={lr} {opt} wd={wd} P={p_size} ===")
        row = run_probe(hidden, epochs, lr, opt, wd, p_size)
        rows.append(row)
        print(row)
    print("\nPROBE SUMMARY")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
