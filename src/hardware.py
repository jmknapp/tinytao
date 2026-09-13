"""GPU sizing. Shrink the population rather than crash."""

from __future__ import annotations

from collections.abc import Callable

import torch

from src.population.mlp import PopulationMLP


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def fit_population_size(
    factory: Callable[[int], PopulationMLP],
    probe_x: torch.Tensor,
    probe_y: torch.Tensor,
    start: int,
    minimum: int,
    steps: int = 2,
) -> int:
    """Binary-backoff on OOM. ``factory(P)`` must allocate a fresh model.

    The probe model is discarded so the real run starts from a clean seed.
    """
    p = int(start)
    minimum = max(1, int(minimum))
    last_err: RuntimeError | None = None
    while p >= minimum:
        model = None
        try:
            if probe_x.device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(probe_x.device)
            model = factory(p)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            for _ in range(steps):
                opt.zero_grad(set_to_none=True)
                logits = model(probe_x)
                b, pop, c = logits.shape
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(b * pop, c),
                    probe_y.unsqueeze(1).expand(b, pop).reshape(-1),
                )
                loss.backward()
                opt.step()
            if probe_x.device.type == "cuda":
                torch.cuda.synchronize(probe_x.device)
            del model, opt
            if probe_x.device.type == "cuda":
                torch.cuda.empty_cache()
            return p
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            last_err = exc
            del model
            if probe_x.device.type == "cuda":
                torch.cuda.empty_cache()
            nxt = max(minimum, p // 2)
            if nxt == p:
                break
            p = nxt
    raise RuntimeError(f"unable to fit population >= {minimum} on device") from last_err
