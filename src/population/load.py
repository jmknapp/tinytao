"""Rebuild a PopulationMLP from a saved slice checkpoint."""

from __future__ import annotations

from pathlib import Path

import torch

from src.population.mlp import PopulationMLP


def load_successful(
    path: str | Path,
    hidden_dims: list[int],
    input_dim: int,
    output_dim: int,
    activation: str = "relu",
    device: torch.device | str = "cpu",
) -> tuple[PopulationMLP, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    n = int(ckpt["W.0"].shape[0])
    model = PopulationMLP(
        population=n,
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        activation=activation,
        device=device,
    )
    with torch.no_grad():
        for i in range(len(model.weights)):
            model.weights[i].copy_(ckpt[f"W.{i}"].to(device))
            if model.biases is not None:
                model.biases[i].copy_(ckpt[f"b.{i}"].to(device))
    meta = {k: v for k, v in ckpt.items() if k not in {f"W.{i}" for i in range(8)} | {f"b.{i}" for i in range(8)}}
    return model, meta
