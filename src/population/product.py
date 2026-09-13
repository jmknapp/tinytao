"""Population of independent product-interaction networks.

Model B: each hidden unit is
    h_j = relu(u_j[a] * v_j[b] + c_j)
The only change from Model A is * instead of +. Forward, loss, and
backward stay fully batched on the population axis.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.population.mlp import ACTIVATIONS, _xavier_uniform_last2_


class PopulationProductMLP(nn.Module):
    """P independent networks with elementwise embedding products.

    U, V: [P, n, H]  maps one-hot residue -> hidden (n = input_dim // 2)
    c:    [P, H]
    W:    [P, H, C]
    b:    [P, C]
    """

    def __init__(
        self,
        population: int,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        activation: str = "relu",
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
            raise ValueError("product model is one hidden layer only")
        if input_dim % 2 != 0:
            raise ValueError("input_dim must be even (concat one-hots)")
        if activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {sorted(ACTIVATIONS)}")
        if not use_bias:
            raise ValueError("product model uses hidden and output bias")

        self.population = int(population)
        self.input_dim = int(input_dim)
        self.n_group = self.input_dim // 2
        self.hidden_dims = [int(h) for h in hidden_dims]
        self.hidden = self.hidden_dims[0]
        self.output_dim = int(output_dim)
        self.activation_name = activation
        self.use_bias = True
        self.init = init
        self.init_scale = float(init_scale)
        self._act = ACTIVATIONS[activation]
        self.widths = [self.input_dim, self.hidden, self.output_dim]

        factory = {"device": device, "dtype": dtype}
        self.U = nn.Parameter(torch.empty(self.population, self.n_group, self.hidden, **factory))
        self.V = nn.Parameter(torch.empty(self.population, self.n_group, self.hidden, **factory))
        self.c = nn.Parameter(torch.empty(self.population, self.hidden, **factory))
        self.W_out = nn.Parameter(torch.empty(self.population, self.hidden, self.output_dim, **factory))
        self.b_out = nn.Parameter(torch.empty(self.population, self.output_dim, **factory))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            for W in (self.U, self.V, self.W_out):
                if self.init == "xavier_uniform":
                    _xavier_uniform_last2_(W)
                else:
                    raise ValueError(f"unknown init {self.init}")
                W.mul_(self.init_scale)
            self.c.zero_()
            self.b_out.zero_()

    def n_params_per_network(self) -> int:
        return 2 * self.n_group * self.hidden + self.hidden + self.hidden * self.output_dim + self.output_dim

    def forward(self, x: torch.Tensor, hidden: bool = False):
        if x.ndim != 2 or x.shape[-1] != self.input_dim:
            raise ValueError(f"expected x [B, {self.input_dim}], got {tuple(x.shape)}")
        n = self.n_group
        ua = torch.einsum("bi,pih->bph", x[:, :n], self.U)
        vb = torch.einsum("bi,pih->bph", x[:, n:], self.V)
        h = self._act(ua * vb + self.c)
        logits = torch.einsum("bph,phc->bpc", h, self.W_out) + self.b_out
        return (logits, [h]) if hidden else logits

    def param_l2(self) -> torch.Tensor:
        acc = (
            self.U.square().sum(dim=(1, 2))
            + self.V.square().sum(dim=(1, 2))
            + self.c.square().sum(dim=1)
            + self.W_out.square().sum(dim=(1, 2))
            + self.b_out.square().sum(dim=1)
        )
        return acc.sqrt()

    def grad_l2(self) -> torch.Tensor:
        acc = torch.zeros(self.population, device=self.U.device, dtype=self.U.dtype)
        for t in (self.U, self.V, self.c, self.W_out, self.b_out):
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
            "U": self.U.detach().cpu()[indices].contiguous(),
            "V": self.V.detach().cpu()[indices].contiguous(),
            "c": self.c.detach().cpu()[indices].contiguous(),
            "W_out": self.W_out.detach().cpu()[indices].contiguous(),
            "b_out": self.b_out.detach().cpu()[indices].contiguous(),
            "indices": indices,
        }

    def describe(self) -> dict:
        return {
            "type": "PopulationProductMLP",
            "interaction": "product",
            "population": self.population,
            "widths": self.widths,
            "activation": self.activation_name,
            "params_per_network": self.n_params_per_network(),
            "dtype": str(self.U.dtype).replace("torch.", ""),
        }


@torch.no_grad()
def sequential_product_logits(model: PopulationProductMLP, x: torch.Tensor, network_index: int) -> torch.Tensor:
    n = model.n_group
    ua = x[:, :n] @ model.U[network_index]
    vb = x[:, n:] @ model.V[network_index]
    h = model._act(ua * vb + model.c[network_index])
    return h @ model.W_out[network_index] + model.b_out[network_index]
