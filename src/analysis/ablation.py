"""Causal probes that do not assume an interpretation."""

from __future__ import annotations

import torch

from src.population.mlp import PopulationMLP
from src.tasks.modular import build_task


@torch.no_grad()
def domain_accuracy(
    model: PopulationMLP,
    p: int,
    device: torch.device | str,
    task_name: str = "modular_multiplication",
) -> torch.Tensor:
    task = build_task(task_name, p)
    _, _, x, y = task.full_domain(device)
    pred = model(x).argmax(dim=-1)
    return (pred == y.unsqueeze(1)).float().mean(dim=0)


@torch.no_grad()
def unit_ablation(
    model: PopulationMLP,
    p: int,
    device: torch.device | str,
    task_name: str = "modular_multiplication",
) -> torch.Tensor:
    """Domain accuracy after zeroing each hidden unit, shape [P, H].

    Implementation: clone W2 and the hidden bias contribution by zeroing
    the corresponding column of W1 / row of W2 / bias entry.
    """
    h = model.hidden_dims[-1]
    base_w1 = model.weights[0].detach().clone()
    base_w2 = model.weights[1].detach().clone()
    base_b1 = model.biases[0].detach().clone() if model.biases is not None else None
    acc = torch.empty(model.population, h, device=str(device))
    for j in range(h):
        with torch.no_grad():
            model.weights[0].copy_(base_w1)
            model.weights[1].copy_(base_w2)
            if base_b1 is not None:
                model.biases[0].copy_(base_b1)
            model.weights[0][:, :, j] = 0
            model.weights[1][:, j, :] = 0
            if model.biases is not None:
                model.biases[0][:, j] = 0
        acc[:, j] = domain_accuracy(model, p, device, task_name=task_name)
    with torch.no_grad():
        model.weights[0].copy_(base_w1)
        model.weights[1].copy_(base_w2)
        if base_b1 is not None:
            model.biases[0].copy_(base_b1)
    return acc


@torch.no_grad()
def residue_shift_accuracy(
    model: PopulationMLP,
    p: int,
    device: torch.device | str,
    shift: int = 1,
    task_name: str = "modular_multiplication",
) -> torch.Tensor:
    """Accuracy after cycling the *a* one-hot rows of W1 by ``shift``.

    A representation that is a single Fourier mode on a is shift-equivariant
    in phase but the readout was trained at one alignment, so exact accuracy
    should collapse unless the rest of the net compensates. This is a
    falsifier for "the network is just a lookup table over raw residue IDs"
    only insofar as a rigid rotation of labels is not the identity; it is
    *not* by itself evidence of a Fourier algorithm.
    """
    W1 = model.weights[0].detach().clone()
    a_rows = W1[:, :p, :].clone()
    W1[:, :p, :] = torch.roll(a_rows, shifts=int(shift), dims=1)
    saved = model.weights[0].detach().clone()
    model.weights[0].copy_(W1)
    acc = domain_accuracy(model, p, device, task_name=task_name)
    model.weights[0].copy_(saved)
    return acc
