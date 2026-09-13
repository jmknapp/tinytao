"""Shrink exact solvers: quantization and hidden-unit removal."""

from __future__ import annotations

import torch

from src.population.mlp import PopulationMLP


@torch.no_grad()
def quantize_inplace(model: PopulationMLP, bits: int) -> None:
    """Per-network symmetric min-max quantization of every parameter tensor.

    ``bits=0`` is a no-op. Each network slice is scaled independently so a
    large-norm organism does not steal range from a small one.
    """
    if bits <= 0:
        return
    levels = (1 << (bits - 1)) - 1
    if levels < 1:
        raise ValueError("bits must be >= 2")
    for tensor in list(model.weights) + (list(model.biases) if model.biases is not None else []):
        # tensor [P, ...]
        flat = tensor.reshape(tensor.shape[0], -1)
        scale = flat.abs().amax(dim=1).clamp_min(1e-12) / levels
        q = torch.round(flat / scale.unsqueeze(1)).clamp(-levels, levels)
        tensor.copy_((q * scale.unsqueeze(1)).reshape_as(tensor))


@torch.no_grad()
def copy_kept_units(
    src: PopulationMLP,
    keep: torch.Tensor,
    device: torch.device | str,
) -> PopulationMLP:
    """Build a narrower population using a subset of hidden units.

    ``keep`` is [P, H_keep] integer indices into the source hidden axis.
    Source must be a 1-hidden-layer MLP.
    """
    if len(src.hidden_dims) != 1:
        raise ValueError("copy_kept_units expects a single hidden layer")
    p, h_keep = keep.shape
    if p != src.population:
        raise ValueError("keep rows must match population")
    dst = PopulationMLP(
        population=p,
        input_dim=src.input_dim,
        hidden_dims=[h_keep],
        output_dim=src.output_dim,
        activation=src.activation_name,
        use_bias=src.use_bias,
        device=device,
    )
    keep = keep.long()
    for i in range(p):
        idx = keep[i]
        dst.weights[0][i].copy_(src.weights[0][i][:, idx])
        dst.weights[1][i].copy_(src.weights[1][i][idx, :])
        if src.biases is not None and dst.biases is not None:
            dst.biases[0][i].copy_(src.biases[0][i][idx])
            dst.biases[1][i].copy_(src.biases[1][i])
    return dst


def least_critical_units(ablation_acc: torch.Tensor, n_drop: int = 1) -> torch.Tensor:
    """Units with the *highest* post-ablation accuracy (safest to drop).

    Returns keep-indices of shape [P, H - n_drop].
    """
    h = ablation_acc.shape[1]
    if n_drop < 1 or n_drop >= h:
        raise ValueError("n_drop must be in 1..H-1")
    order = torch.argsort(ablation_acc, dim=1, descending=True)
    drop = order[:, :n_drop]
    keep = []
    for i in range(ablation_acc.shape[0]):
        all_u = torch.arange(h)
        mask = torch.ones(h, dtype=torch.bool)
        mask[drop[i]] = False
        keep.append(all_u[mask])
    return torch.stack(keep, dim=0)
