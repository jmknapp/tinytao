"""Vectorized training of a population on one shared dataset."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.population.mlp import PopulationMLP


@dataclass
class TrainResult:
    epochs_run: int
    log_epochs: list[int]
    loss: torch.Tensor
    train_acc: torch.Tensor
    test_acc: torch.Tensor
    domain_acc: torch.Tensor
    param_norm: torch.Tensor
    grad_norm: torch.Tensor
    hidden_mean: torch.Tensor
    hidden_sparsity: torch.Tensor
    final_train_acc: torch.Tensor
    final_test_acc: torch.Tensor
    final_domain_acc: torch.Tensor
    final_loss: torch.Tensor
    seconds: float
    step_ms: float
    examples_per_sec: float
    network_evals_per_sec: float
    peak_vram_bytes: int
    extra: dict[str, Any] = field(default_factory=dict)


def _optimizer(model: PopulationMLP, name: str, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    key = name.lower()
    if key == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if key == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if key == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    raise ValueError(f"unknown optimizer {name}")


def _accuracy(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Per-network accuracy. logits [B, P, C], y [B] -> [P]."""
    pred = logits.argmax(dim=-1)
    return (pred == y.unsqueeze(1)).float().mean(dim=0)


def _cross_entropy(logits: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean CE plus per-network mean CE. logits [B, P, C], y [B]."""
    b, p, c = logits.shape
    per = F.cross_entropy(
        logits.reshape(b * p, c),
        y.unsqueeze(1).expand(b, p).reshape(-1),
        reduction="none",
    ).reshape(b, p)
    per_net = per.mean(dim=0)
    return per.mean(), per_net


def _hidden_stats(acts: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean activation and fraction <= 0, averaged over batch and units -> [P]."""
    if not acts:
        raise ValueError("no hidden activations")
    h = acts[-1]
    return h.mean(dim=(0, 2)), (h <= 0).float().mean(dim=(0, 2))


@torch.no_grad()
def evaluate(
    model: PopulationMLP,
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (mean_loss [P], accuracy [P]) on a fixed set."""
    model.eval()
    if batch_size <= 0 or batch_size >= x.shape[0]:
        logits = model(x)
        loss, per_net = _cross_entropy(logits, y)
        _ = loss
        return per_net, _accuracy(logits, y)

    losses = []
    acc_hits = []
    counts = []
    for start in range(0, x.shape[0], batch_size):
        xb = x[start : start + batch_size]
        yb = y[start : start + batch_size]
        logits = model(xb)
        _, per_net = _cross_entropy(logits, yb)
        losses.append(per_net * xb.shape[0])
        acc_hits.append(_accuracy(logits, yb) * xb.shape[0])
        counts.append(xb.shape[0])
    n = float(sum(counts))
    return sum(losses) / n, sum(acc_hits) / n


def train_population(
    model: PopulationMLP,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    domain_x: torch.Tensor,
    domain_y: torch.Tensor,
    epochs: int,
    lr: float,
    optimizer_name: str = "adam",
    weight_decay: float = 0.0,
    batch_size: int = 0,
    log_every: int = 10,
    eval_every: int = 10,
    grad_clip: float | None = None,
    show_progress: bool = True,
    on_log_step: Any | None = None,
) -> TrainResult:
    device = train_x.device
    p = model.population
    n_train = train_x.shape[0]
    use_full_batch = batch_size <= 0 or batch_size >= n_train

    opt = _optimizer(model, optimizer_name, lr, weight_decay)

    log_epochs: list[int] = []

    # Collect on CPU to keep VRAM for the compute graph.
    hist = {
        "loss": [],
        "train_acc": [],
        "test_acc": [],
        "domain_acc": [],
        "param_norm": [],
        "grad_norm": [],
        "hidden_mean": [],
        "hidden_sparsity": [],
    }

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    t0 = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    t1 = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    if t0 is not None:
        t0.record()
    else:
        import time

        wall0 = time.perf_counter()

    steps = 0
    iterator = range(1, epochs + 1)
    if show_progress:
        iterator = tqdm(iterator, desc="train", leave=False)

    for epoch in iterator:
        model.train()
        if use_full_batch:
            opt.zero_grad(set_to_none=True)
            logits, acts = model(train_x, hidden=True)
            loss, _ = _cross_entropy(logits, train_y)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            steps += 1
            last_loss = loss.detach()
            last_acts = [a.detach() for a in acts]
            last_train_acc = _accuracy(logits.detach(), train_y)
        else:
            perm = torch.randperm(n_train, device=device)
            last_loss = None
            last_acts = None
            correct = torch.zeros(p, device=device)
            seen = 0
            for start in range(0, n_train, batch_size):
                idx = perm[start : start + batch_size]
                xb, yb = train_x[idx], train_y[idx]
                opt.zero_grad(set_to_none=True)
                logits, acts = model(xb, hidden=True)
                loss, _ = _cross_entropy(logits, yb)
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
                steps += 1
                last_loss = loss.detach()
                last_acts = [a.detach() for a in acts]
                correct += (logits.detach().argmax(-1) == yb.unsqueeze(1)).float().sum(0)
                seen += xb.shape[0]
            last_train_acc = correct / seen

        on_log = epoch % log_every == 0 or epoch == epochs
        on_eval = epoch % eval_every == 0 or epoch == epochs
        if on_log:
            log_epochs.append(epoch)
            assert last_loss is not None and last_acts is not None
            h_mean, h_sparse = _hidden_stats(last_acts)
            with torch.no_grad():
                if use_full_batch:
                    _, per_net_loss = _cross_entropy(model(train_x), train_y)
                else:
                    per_net_loss = last_loss.expand(p)
            hist["loss"].append(per_net_loss.detach().cpu())
            hist["train_acc"].append(last_train_acc.detach().cpu())
            hist["param_norm"].append(model.param_l2().detach().cpu())
            hist["grad_norm"].append(model.grad_l2().detach().cpu())
            hist["hidden_mean"].append(h_mean.detach().cpu())
            hist["hidden_sparsity"].append(h_sparse.detach().cpu())
            if on_eval:
                _, test_acc = evaluate(model, test_x, test_y)
                _, domain_acc = evaluate(model, domain_x, domain_y)
            else:
                test_acc = hist["test_acc"][-1] if hist["test_acc"] else torch.zeros(p)
                domain_acc = hist["domain_acc"][-1] if hist["domain_acc"] else torch.zeros(p)
            hist["test_acc"].append(test_acc.detach().cpu())
            hist["domain_acc"].append(domain_acc.detach().cpu())
            if on_log_step is not None:
                on_log_step(
                    epoch,
                    {
                        "model": model,
                        "train_acc": hist["train_acc"][-1],
                        "test_acc": hist["test_acc"][-1],
                        "domain_acc": hist["domain_acc"][-1],
                    },
                )
            if show_progress and hasattr(iterator, "set_postfix"):
                iterator.set_postfix(
                    loss=float(hist["loss"][-1].mean()),
                    train=float(hist["train_acc"][-1].mean()),
                    domain=float(hist["domain_acc"][-1].mean()),
                    exact=int((hist["domain_acc"][-1] >= 1.0 - 1e-12).sum()),
                )

    if t1 is not None:
        t1.record()
        torch.cuda.synchronize(device)
        seconds = t0.elapsed_time(t1) / 1000.0
        peak = int(torch.cuda.max_memory_allocated(device))
    else:
        import time

        seconds = time.perf_counter() - wall0
        peak = 0

    def stack(key: str) -> torch.Tensor:
        return torch.stack(hist[key], dim=1) if hist[key] else torch.empty(p, 0)

    # Final exhaustive metrics (not just the last logged train batch).
    with torch.no_grad():
        final_train_loss, final_train_acc = evaluate(model, train_x, train_y)
        _, final_test_acc = evaluate(model, test_x, test_y)
        _, final_domain_acc = evaluate(model, domain_x, domain_y)

    examples = steps * (n_train if use_full_batch else min(batch_size, n_train))
    # Rough: last minibatch size varies; use train-set size * epochs as the
    # scientific throughput (each epoch presents the full train set once).
    examples = epochs * n_train
    evs = examples * p

    return TrainResult(
        epochs_run=epochs,
        log_epochs=log_epochs,
        loss=stack("loss"),
        train_acc=stack("train_acc"),
        test_acc=stack("test_acc"),
        domain_acc=stack("domain_acc"),
        param_norm=stack("param_norm"),
        grad_norm=stack("grad_norm"),
        hidden_mean=stack("hidden_mean"),
        hidden_sparsity=stack("hidden_sparsity"),
        final_train_acc=final_train_acc.detach().cpu(),
        final_test_acc=final_test_acc.detach().cpu(),
        final_domain_acc=final_domain_acc.detach().cpu(),
        final_loss=final_train_loss.detach().cpu(),
        seconds=seconds,
        step_ms=1000.0 * seconds / max(epochs, 1),
        examples_per_sec=examples / max(seconds, 1e-9),
        network_evals_per_sec=evs / max(seconds, 1e-9),
        peak_vram_bytes=peak,
        extra={"optimizer_steps": steps, "full_batch": use_full_batch},
    )
