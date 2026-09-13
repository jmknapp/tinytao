"""Population bookkeeping. No mechanistic claims."""

from __future__ import annotations

from typing import Any

import torch

from src.population.mlp import PopulationMLP
from src.training.metrics import OutcomeCounts, detect_transitions


LABEL_NAMES = {0: "fail", 1: "partial", 2: "memorize", 3: "generalize", 4: "exact"}


def summarize_population(
    labels: torch.Tensor,
    counts: OutcomeCounts,
    train_acc: torch.Tensor,
    test_acc: torch.Tensor,
    domain_acc: torch.Tensor,
    param_norm: torch.Tensor,
    domain_acc_traj: torch.Tensor,
    n_classes: int,
) -> dict[str, Any]:
    trans = detect_transitions(domain_acc_traj)
    exact = labels == 4
    return {
        "counts": counts.as_dict(),
        "fractions": counts.fractions(),
        "n_classes": n_classes,
        "mean_train_acc": float(train_acc.mean()),
        "mean_test_acc": float(test_acc.mean()),
        "mean_domain_acc": float(domain_acc.mean()),
        "median_domain_acc": float(domain_acc.median()),
        "max_domain_acc": float(domain_acc.max()),
        "n_sudden_transition": int(trans["sudden_transition"].sum()),
        "n_reached_99": int((trans["first_exact_log"] >= 0).sum()),
        "exact_param_norm_mean": float(param_norm[exact].mean()) if exact.any() else None,
        "exact_param_norm_std": float(param_norm[exact].std(unbiased=False)) if exact.any() else None,
    }


def select_representatives(
    model: PopulationMLP,
    labels: torch.Tensor,
    domain_acc: torch.Tensor,
    sudden: torch.Tensor,
    max_exact: int = 64,
    per_bin: int = 4,
) -> dict[str, torch.Tensor]:
    """Keep a small, diverse set of final weights.

    Exact solvers are the scientifically valuable objects; we cap how many
    we store. Other bins contribute a few examples so failures remain
    inspectable.
    """
    chosen: list[int] = []
    for lab in (4, 3, 2, 1, 0):
        idx = torch.where(labels == lab)[0]
        if idx.numel() == 0:
            continue
        cap = max_exact if lab == 4 else per_bin
        if lab == 4:
            # Prefer sudden transitions, then highest domain acc (all 1).
            score = sudden[idx].float() * 10 + domain_acc[idx]
            order = torch.argsort(score, descending=True)
        else:
            order = torch.argsort(domain_acc[idx], descending=True)
        pick = idx[order[:cap]].tolist()
        chosen.extend(pick)

    # Always include the best and worst complete-domain networks.
    chosen.append(int(domain_acc.argmax()))
    chosen.append(int(domain_acc.argmin()))
    uniq = sorted(set(chosen))
    state = model.slice_state_dict(uniq)
    state["labels"] = labels[uniq].cpu()
    state["domain_acc"] = domain_acc[uniq].cpu()
    return state
