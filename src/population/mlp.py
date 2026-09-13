"""Population of independent MLPs stored as extra tensor dimensions.

A Python loop over networks is used only for correctness tests and for
initialization that must treat each slice as its own matrix. Forward,
loss, and backward are fully batched.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


ACTIVATIONS = {
    "relu": F.relu,
    "gelu": F.gelu,
    "tanh": torch.tanh,
    "silu": F.silu,
    "identity": lambda x: x,
}


def _xavier_uniform_last2_(tensor: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
    """Xavier init using only the last two axes as (fan_in, fan_out).

    ``nn.init.xavier_uniform_`` on a [P, in, out] tensor treats P as a
    feature dimension and produces the wrong scale. Each network slice
    must be initialized as an ordinary matrix.
    """
    fan_in = tensor.shape[-2]
    fan_out = tensor.shape[-1]
    bound = gain * math.sqrt(6.0 / (fan_in + fan_out))
    return tensor.uniform_(-bound, bound)


class PopulationMLP(nn.Module):
    """P independent MLPs sharing architecture, not parameters.

    Layer ``i`` maps width[i] -> width[i+1] with
        W[i]: [P, width[i], width[i+1]]
        b[i]: [P, width[i+1]]
    """

    def __init__(
        self,
        population: int,
        input_dim: int,
        hidden_dims: Sequence[int],
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
        if activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {sorted(ACTIVATIONS)}")

        self.population = int(population)
        self.input_dim = int(input_dim)
        self.hidden_dims = [int(h) for h in hidden_dims]
        self.output_dim = int(output_dim)
        self.activation_name = activation
        self.use_bias = bool(use_bias)
        self.init = init
        self.init_scale = float(init_scale)
        self._act = ACTIVATIONS[activation]

        widths = [self.input_dim, *self.hidden_dims, self.output_dim]
        if any(w < 1 for w in widths):
            raise ValueError(f"all layer widths must be positive, got {widths}")
        self.widths = widths

        factory = {"device": device, "dtype": dtype}
        self.weights = nn.ParameterList(
            [
                nn.Parameter(torch.empty(self.population, din, dout, **factory))
                for din, dout in zip(widths[:-1], widths[1:])
            ]
        )
        if self.use_bias:
            self.biases = nn.ParameterList(
                [nn.Parameter(torch.empty(self.population, dout, **factory)) for dout in widths[1:]]
            )
        else:
            self.biases = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            for W in self.weights:
                if self.init == "xavier_uniform":
                    _xavier_uniform_last2_(W)
                elif self.init == "kaiming_uniform":
                    bound = math.sqrt(6.0 / W.shape[-2])
                    W.uniform_(-bound, bound)
                elif self.init == "orthogonal":
                    for i in range(self.population):
                        nn.init.orthogonal_(W[i])
                else:
                    raise ValueError(f"unknown init {self.init}")
                W.mul_(self.init_scale)
            if self.biases is not None:
                for b in self.biases:
                    b.zero_()

    def n_params_per_network(self) -> int:
        w = sum(din * dout for din, dout in zip(self.widths[:-1], self.widths[1:]))
        b = sum(self.widths[1:]) if self.use_bias else 0
        return w + b

    def forward(self, x: torch.Tensor, hidden: bool = False) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Evaluate every network on the same batch.

        Parameters
        ----------
        x:
            [B, input_dim]
        hidden:
            If True, also return pre-output activations for each hidden layer.

        Returns
        -------
        logits : [B, P, output_dim]
        hiddens : list of [B, P, width] if ``hidden``
        """
        if x.ndim != 2 or x.shape[-1] != self.input_dim:
            raise ValueError(f"expected x [B, {self.input_dim}], got {tuple(x.shape)}")

        h = x
        acts: list[torch.Tensor] = []
        last = len(self.weights) - 1
        for i, W in enumerate(self.weights):
            # h @ W_p + b_p  for each network p, without a Python loop.
            if h.ndim == 2:
                h = torch.einsum("bi,pih->bph", h, W)
            else:
                h = torch.einsum("bpi,pih->bph", h, W)
            if self.biases is not None:
                h = h + self.biases[i]
            if i != last:
                h = self._act(h)
                acts.append(h)
        return (h, acts) if hidden else h

    def param_l2(self) -> torch.Tensor:
        """Per-network parameter L2 norm, shape [P]."""
        acc = torch.zeros(self.population, device=self.weights[0].device, dtype=self.weights[0].dtype)
        for W in self.weights:
            acc = acc + W.square().sum(dim=(1, 2))
        if self.biases is not None:
            for b in self.biases:
                acc = acc + b.square().sum(dim=1)
        return acc.sqrt()

    def grad_l2(self) -> torch.Tensor:
        """Per-network gradient L2 norm, shape [P]. Zeros if grads are missing."""
        acc = torch.zeros(self.population, device=self.weights[0].device, dtype=self.weights[0].dtype)
        tensors: list[torch.Tensor] = list(self.weights)
        if self.biases is not None:
            tensors.extend(self.biases)
        for t in tensors:
            if t.grad is None:
                continue
            dims = tuple(range(1, t.grad.ndim))
            acc = acc + t.grad.square().sum(dim=dims)
        return acc.sqrt()

    def slice_state_dict(self, indices: torch.Tensor | list[int]) -> dict[str, torch.Tensor]:
        """CPU copies of parameters for a subset of networks."""
        if not torch.is_tensor(indices):
            indices = torch.as_tensor(indices, dtype=torch.long)
        indices = indices.detach().cpu().long()
        out: dict[str, torch.Tensor] = {}
        for i, W in enumerate(self.weights):
            out[f"W.{i}"] = W.detach().cpu()[indices].contiguous()
        if self.biases is not None:
            for i, b in enumerate(self.biases):
                out[f"b.{i}"] = b.detach().cpu()[indices].contiguous()
        out["indices"] = indices
        return out

    def describe(self) -> dict:
        return {
            "type": "PopulationMLP",
            "population": self.population,
            "widths": self.widths,
            "activation": self.activation_name,
            "use_bias": self.use_bias,
            "init": self.init,
            "init_scale": self.init_scale,
            "params_per_network": self.n_params_per_network(),
            "dtype": str(self.weights[0].dtype).replace("torch.", ""),
        }


@torch.no_grad()
def sequential_reference_logits(
    model: PopulationMLP, x: torch.Tensor, network_index: int
) -> torch.Tensor:
    """Slow per-network forward used only to test the batched implementation."""
    h = x
    last = len(model.weights) - 1
    for i, W in enumerate(model.weights):
        h = h @ W[network_index] + (model.biases[i][network_index] if model.biases is not None else 0)
        if i != last:
            h = model._act(h)
    return h
