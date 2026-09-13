"""Population of independent complex-composition networks.

Model C: one shared 2-D embedding table e(a) = (x_a, y_a) and the
interaction of complex multiplication

    (x_a x_b - y_a y_b,  x_a y_b + y_a x_b).

The architecture supplies z(a) z(b). It does not supply the discrete
log that would make e(a) a root of unity. Several independent heads
may be concatenated; each head is its own 2-D embedding table.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.population.mlp import _xavier_uniform_last2_


class PopulationComplexMLP(nn.Module):
    """P independent networks with complex-multiplication interaction.

    E:     [P, n, H, 2]  shared residue embedding (used for both a and b)
    W_out: [P, 2H, C]
    b_out: [P, C]
    """

    def __init__(
        self,
        population: int,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
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
        if len(hidden_dims) != 1:
            raise ValueError("complex model is one composition layer only")
        if input_dim % 2 != 0:
            raise ValueError("input_dim must be even (concat one-hots)")
        if hidden_dims[0] < 1:
            raise ValueError("need at least one complex head")

        self.population = int(population)
        self.input_dim = int(input_dim)
        self.n_group = self.input_dim // 2
        self.hidden_dims = [int(h) for h in hidden_dims]
        self.heads = self.hidden_dims[0]
        self.output_dim = int(output_dim)
        self.activation_name = "identity"
        self.use_bias = True
        self.init = init
        self.init_scale = float(init_scale)
        self.widths = [self.input_dim, 2 * self.heads, self.output_dim]
        _ = activation, use_bias

        factory = {"device": device, "dtype": dtype}
        self.E = nn.Parameter(
            torch.empty(self.population, self.n_group, self.heads, 2, **factory)
        )
        self.W_out = nn.Parameter(
            torch.empty(self.population, 2 * self.heads, self.output_dim, **factory)
        )
        self.b_out = nn.Parameter(torch.empty(self.population, self.output_dim, **factory))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            flat = self.E.reshape(self.population, self.n_group, 2 * self.heads)
            if self.init == "xavier_uniform":
                _xavier_uniform_last2_(flat)
                _xavier_uniform_last2_(self.W_out)
            else:
                raise ValueError(f"unknown init {self.init}")
            self.E.mul_(self.init_scale)
            self.W_out.mul_(self.init_scale)
            self.b_out.zero_()

    def n_params_per_network(self) -> int:
        return (
            self.n_group * self.heads * 2
            + (2 * self.heads) * self.output_dim
            + self.output_dim
        )

    def compose(self, x: torch.Tensor) -> torch.Tensor:
        """Return concatenated complex products, shape [B, P, 2H]."""
        if x.ndim != 2 or x.shape[-1] != self.input_dim:
            raise ValueError(f"expected x [B, {self.input_dim}], got {tuple(x.shape)}")
        n = self.n_group
        xa = torch.einsum("bi,pihd->bphd", x[:, :n], self.E)
        xb = torch.einsum("bi,pihd->bphd", x[:, n:], self.E)
        real = xa[..., 0] * xb[..., 0] - xa[..., 1] * xb[..., 1]
        imag = xa[..., 0] * xb[..., 1] + xa[..., 1] * xb[..., 0]
        return torch.stack((real, imag), dim=-1).reshape(x.shape[0], self.population, 2 * self.heads)

    def forward(self, x: torch.Tensor, hidden: bool = False):
        h = self.compose(x)
        logits = torch.einsum("bpk,pkc->bpc", h, self.W_out) + self.b_out
        return (logits, [h]) if hidden else logits

    def param_l2(self) -> torch.Tensor:
        acc = (
            self.E.square().sum(dim=(1, 2, 3))
            + self.W_out.square().sum(dim=(1, 2))
            + self.b_out.square().sum(dim=1)
        )
        return acc.sqrt()

    def grad_l2(self) -> torch.Tensor:
        acc = torch.zeros(self.population, device=self.E.device, dtype=self.E.dtype)
        for t in (self.E, self.W_out, self.b_out):
            if t.grad is None:
                continue
            dims = tuple(range(1, t.grad.ndim))
            acc = acc + t.grad.square().sum(dim=dims)
        return acc.sqrt()

    def slice_state_dict(self, indices: torch.Tensor | list[int]) -> dict[str, torch.Tensor]:
        if not torch.is_tensor(indices):
            indices = torch.as_tensor(indices, dtype=torch.long)
        indices = indices.detach().cpu().long()
        return {
            "E": self.E.detach().cpu()[indices].contiguous(),
            "W_out": self.W_out.detach().cpu()[indices].contiguous(),
            "b_out": self.b_out.detach().cpu()[indices].contiguous(),
            "indices": indices,
        }

    def describe(self) -> dict:
        return {
            "type": "PopulationComplexMLP",
            "interaction": "complex",
            "population": self.population,
            "heads": self.heads,
            "widths": self.widths,
            "activation": self.activation_name,
            "shared_embedding": True,
            "params_per_network": self.n_params_per_network(),
            "dtype": str(self.E.dtype).replace("torch.", ""),
        }


@torch.no_grad()
def sequential_complex_logits(model: PopulationComplexMLP, x: torch.Tensor, network_index: int) -> torch.Tensor:
    n = model.n_group
    e = model.E[network_index]
    xa = torch.einsum("bi,ihd->bhd", x[:, :n], e)
    xb = torch.einsum("bi,ihd->bhd", x[:, n:], e)
    real = xa[..., 0] * xb[..., 0] - xa[..., 1] * xb[..., 1]
    imag = xa[..., 0] * xb[..., 1] + xa[..., 1] * xb[..., 0]
    h = torch.stack((real, imag), dim=-1).reshape(x.shape[0], 2 * model.heads)
    return h @ model.W_out[network_index] + model.b_out[network_index]
