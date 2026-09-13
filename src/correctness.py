"""Prove the batched population matches independent networks."""

from __future__ import annotations

import json

import torch

from src.population.mlp import PopulationMLP, sequential_reference_logits
from src.tasks.modular import ModularMultiplication


def check_forward_matches_reference(
    device: torch.device | str = "cpu",
    atol: float = 1e-5,
) -> dict[str, float]:
    torch.manual_seed(0)
    task = ModularMultiplication(p=13)
    _, _, x, _ = task.full_domain(device)
    x = x[:32]
    model = PopulationMLP(
        population=7,
        input_dim=task.input_dim,
        hidden_dims=[5, 4],
        output_dim=task.output_dim,
        activation="relu",
        device=device,
    )
    batched = model(x)
    worst = 0.0
    for i in range(model.population):
        ref = sequential_reference_logits(model, x, i)
        err = float((batched[:, i, :] - ref).abs().max().detach())
        worst = max(worst, err)
        if err > atol:
            raise AssertionError(f"network {i} forward mismatch max_abs={err}")
    return {"max_abs_forward": worst, "population": model.population, "batch": int(x.shape[0])}


def check_gradients_do_not_mix(device: torch.device | str = "cpu") -> dict[str, float]:
    """Loss on network 0 must leave every other slice's gradient at 0."""
    torch.manual_seed(1)
    task = ModularMultiplication(p=13)
    _, _, x, y = task.full_domain(device)
    x, y = x[:16], y[:16]
    model = PopulationMLP(
        population=5,
        input_dim=task.input_dim,
        hidden_dims=[6],
        output_dim=task.output_dim,
        device=device,
    )
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits[:, 0, :], y)
    loss.backward()
    leaked = 0.0
    for W in model.weights:
        leaked = max(leaked, float(W.grad[1:].abs().max()))
        own = float(W.grad[0].abs().max())
        if own == 0:
            raise AssertionError("network 0 has zero weight gradient")
    if model.biases is not None:
        for b in model.biases:
            leaked = max(leaked, float(b.grad[1:].abs().max()))
    if leaked > 0:
        raise AssertionError(f"gradient leaked across networks: {leaked}")
    return {"max_abs_leak": leaked}


def check_task_table() -> dict[str, int]:
    task = ModularMultiplication(p=13)
    a, b, _, y = task.full_domain()
    expect = (a * b) % 13
    if not torch.equal(y, expect):
        raise AssertionError("modular multiplication table is wrong")
    if y.numel() != 169:
        raise AssertionError("domain is not fully enumerated")
    return {"n_domain": int(y.numel()), "p": 13}


def run_all(device: torch.device | str = "cpu") -> dict:
    return {
        "task_table": check_task_table(),
        "forward": check_forward_matches_reference(device),
        "grad_isolation": check_gradients_do_not_mix(device),
    }


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(json.dumps(run_all(dev), indent=2))
