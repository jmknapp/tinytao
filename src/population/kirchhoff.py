"""Population of Kirchhoff cofactor networks.

Model K: learned per-edge affine weights, Laplacian from those weights,
ĥ = det(L[1:,1:]), logits_k = -(ĥ - k)^2. The architecture supplies the
matrix-tree cofactor. It does not supply 0-1 adjacency.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.population.mlp import _xavier_uniform_last2_


def laplacian_from_weights(w: torch.Tensor, elist: tuple[tuple[int, int], ...], n: int) -> torch.Tensor:
    """w [B, P, E] -> L [B, P, n, n]."""
    b, p, _ = w.shape
    lap = w.new_zeros(b, p, n, n)
    for e, (i, j) in enumerate(elist):
        we = w[:, :, e]
        lap[:, :, i, i] = lap[:, :, i, i] + we
        lap[:, :, j, j] = lap[:, :, j, j] + we
        lap[:, :, i, j] = lap[:, :, i, j] - we
        lap[:, :, j, i] = lap[:, :, j, i] - we
    return lap


class PopulationKirchhoff(nn.Module):
    """P independent Kirchhoff nets. scale,bias: [P, E]."""

    def __init__(
        self,
        population: int,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        n_vertices: int,
        elist: tuple[tuple[int, int], ...],
        activation: str = "identity",
        use_bias: bool = True,
        init: str = "xavier_uniform",
        init_scale: float = 1.0,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if population < 1:
            raise ValueError("population must be >= 1")
        if input_dim != len(elist):
            raise ValueError("input_dim must equal the number of edges")
        _ = hidden_dims, activation, use_bias

        self.population = int(population)
        self.input_dim = int(input_dim)
        self.n_vertices = int(n_vertices)
        self.elist = tuple(elist)
        self.heads = self.input_dim
        self.output_dim = int(output_dim)
        self.activation_name = "identity"
        self.init = init
        self.init_scale = float(init_scale)
        self.hidden_dims = [self.input_dim]
        self.widths = [self.input_dim, 1, self.output_dim]

        factory = {"device": device, "dtype": dtype}
        self.scale = nn.Parameter(torch.empty(self.population, self.input_dim, **factory))
        self.bias = nn.Parameter(torch.empty(self.population, self.input_dim, **factory))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            if self.init == "xavier_uniform":
                flat = self.scale.unsqueeze(-1)
                _xavier_uniform_last2_(flat)
                self.scale.copy_(flat.squeeze(-1))
            else:
                bound = math.sqrt(3.0)
                self.scale.uniform_(-bound, bound)
            self.scale.mul_(self.init_scale)
            self.bias.zero_()

    def n_params_per_network(self) -> int:
        return 2 * self.input_dim

    def edge_weights(self, x: torch.Tensor) -> torch.Tensor:
        """x [B, E] -> w [B, P, E]."""
        return self.scale.unsqueeze(0) * x.unsqueeze(1) + self.bias.unsqueeze(0)

    def cofactor(self, w: torch.Tensor) -> torch.Tensor:
        lap = laplacian_from_weights(w, self.elist, self.n_vertices)
        return torch.linalg.det(lap[:, :, 1:, 1:])

    def forward(self, x: torch.Tensor, hidden: bool = False):
        if x.ndim != 2 or x.shape[-1] != self.input_dim:
            raise ValueError(f"expected x [B, {self.input_dim}], got {tuple(x.shape)}")
        w = self.edge_weights(x)
        hat = self.cofactor(w)
        k = torch.arange(self.output_dim, device=x.device, dtype=hat.dtype)
        logits = -(hat.unsqueeze(-1) - k) ** 2
        return (logits, [w]) if hidden else logits

    def param_l2(self) -> torch.Tensor:
        return (self.scale.square().sum(dim=1) + self.bias.square().sum(dim=1)).sqrt()

    def grad_l2(self) -> torch.Tensor:
        acc = torch.zeros(self.population, device=self.scale.device, dtype=self.scale.dtype)
        for t in (self.scale, self.bias):
            if t.grad is None:
                continue
            acc = acc + t.grad.square().sum(dim=1)
        return acc.sqrt()

    def slice_state_dict(self, indices: torch.Tensor | list[int]) -> dict[str, torch.Tensor]:
        if not torch.is_tensor(indices):
            indices = torch.as_tensor(indices, dtype=torch.long)
        indices = indices.detach().cpu().long()
        return {
            "scale": self.scale.detach().cpu()[indices].contiguous(),
            "bias": self.bias.detach().cpu()[indices].contiguous(),
            "indices": indices,
        }

    def describe(self) -> dict:
        return {
            "type": "PopulationKirchhoff",
            "interaction": "kirchhoff",
            "population": self.population,
            "n_vertices": self.n_vertices,
            "n_edges": self.input_dim,
            "widths": self.widths,
            "activation": self.activation_name,
            "params_per_network": self.n_params_per_network(),
            "dtype": str(self.scale.dtype).replace("torch.", ""),
        }
