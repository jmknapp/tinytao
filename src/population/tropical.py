"""Population of independent min-plus (tropical) networks.

Model T: one shared real table e(a) in R^H and the interaction of
coordinatewise minimum

    h = min(e(a), e(b)).

The architecture supplies a lattice meet. It does not supply integer
valuations. Several independent heads are concatenated; each head is
one real coordinate.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.population.mlp import _xavier_uniform_last2_


class PopulationTropicalMLP(nn.Module):
    """P independent networks with min interaction.

    E:     [P, n, H]
    W_out: [P, H, C]
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
            raise ValueError("tropical model is one composition layer only")
        if input_dim % 2 != 0:
            raise ValueError("input_dim must be even (concat one-hots)")
        if hidden_dims[0] < 1:
            raise ValueError("need at least one tropical head")

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
        self.widths = [self.input_dim, self.heads, self.output_dim]
        _ = activation, use_bias

        factory = {"device": device, "dtype": dtype}
        self.E = nn.Parameter(
            torch.empty(self.population, self.n_group, self.heads, **factory)
        )
        self.W_out = nn.Parameter(
            torch.empty(self.population, self.heads, self.output_dim, **factory)
        )
        self.b_out = nn.Parameter(torch.empty(self.population, self.output_dim, **factory))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            if self.init == "xavier_uniform":
                _xavier_uniform_last2_(self.E)
                _xavier_uniform_last2_(self.W_out)
            else:
                raise ValueError(f"unknown init {self.init}")
            self.E.mul_(self.init_scale)
            self.W_out.mul_(self.init_scale)
            self.b_out.zero_()

    def n_params_per_network(self) -> int:
        return self.n_group * self.heads + self.heads * self.output_dim + self.output_dim

    def compose(self, x: torch.Tensor) -> torch.Tensor:
        """Return coordinatewise minima, shape [B, P, H]."""
        if x.ndim != 2 or x.shape[-1] != self.input_dim:
            raise ValueError(f"expected x [B, {self.input_dim}], got {tuple(x.shape)}")
        n = self.n_group
        xa = torch.einsum("bi,pih->bph", x[:, :n], self.E)
        xb = torch.einsum("bi,pih->bph", x[:, n:], self.E)
        return torch.minimum(xa, xb)

    def forward(self, x: torch.Tensor, hidden: bool = False):
        h = self.compose(x)
        logits = torch.einsum("bph,phc->bpc", h, self.W_out) + self.b_out
        return (logits, [h]) if hidden else logits

    def param_l2(self) -> torch.Tensor:
        acc = (
            self.E.square().sum(dim=(1, 2))
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
            "type": "PopulationTropicalMLP",
            "interaction": "tropical",
            "population": self.population,
            "heads": self.heads,
            "widths": self.widths,
            "activation": self.activation_name,
            "shared_embedding": True,
            "params_per_network": self.n_params_per_network(),
            "dtype": str(self.E.dtype).replace("torch.", ""),
        }
