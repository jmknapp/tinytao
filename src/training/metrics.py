"""Outcome labels and transition detectors.

Observations live here. Interpretations do not.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


# Chance for a p-way classifier. Used only as a reference line.
def chance(p: int) -> float:
    return 1.0 / float(p)


@dataclass(frozen=True)
class OutcomeCounts:
    """Hard labels for each network. Thresholds are documented, not learned."""

    exact: int
    generalize: int
    memorize: int
    partial: int
    fail: int
    total: int

    def as_dict(self) -> dict[str, int]:
        return {
            "exact": self.exact,
            "generalize": self.generalize,
            "memorize": self.memorize,
            "partial": self.partial,
            "fail": self.fail,
            "total": self.total,
        }

    def fractions(self) -> dict[str, float]:
        n = max(self.total, 1)
        return {k: v / n for k, v in self.as_dict().items() if k != "total"}


def classify_outcomes(
    train_acc: torch.Tensor,
    test_acc: torch.Tensor,
    domain_acc: torch.Tensor,
    n_classes: int,
    exact_min: float = 1.0,
    generalize_test: float = 0.90,
    memorize_train: float = 0.99,
    memorize_gap: float = 0.20,
    fail_max: float = 0.0,
) -> tuple[torch.Tensor, OutcomeCounts]:
    """Assign one mutually exclusive label per network.

    Labels (first match wins):
        exact       domain_acc == 1 (or >= exact_min; default requires every input)
        memorize    train almost perfect, test worse by ``memorize_gap``
        generalize  test_acc >= generalize_test but not exact on the domain
        fail        domain_acc <= chance + fail_max slack (default: at or below chance)
        partial     everything else

    ``exact`` is the only scientifically privileged class: the network is a
    solver of the fully enumerated problem. The other labels are operational
    bins for population bookkeeping.
    """
    if fail_max == 0.0:
        fail_ceiling = chance(n_classes) + 1e-6
    else:
        fail_ceiling = chance(n_classes) + fail_max

    exact = domain_acc >= exact_min
    memorize = (~exact) & (train_acc >= memorize_train) & ((train_acc - test_acc) >= memorize_gap)
    generalize = (~exact) & (~memorize) & (test_acc >= generalize_test)
    fail = (~exact) & (~memorize) & (~generalize) & (domain_acc <= fail_ceiling)
    partial = ~(exact | memorize | generalize | fail)

    labels = torch.zeros(train_acc.shape[0], dtype=torch.int64)
    labels[fail] = 0
    labels[partial] = 1
    labels[memorize] = 2
    labels[generalize] = 3
    labels[exact] = 4

    counts = OutcomeCounts(
        exact=int(exact.sum()),
        generalize=int(generalize.sum()),
        memorize=int(memorize.sum()),
        partial=int(partial.sum()),
        fail=int(fail.sum()),
        total=int(train_acc.numel()),
    )
    return labels, counts


def detect_transitions(
    domain_acc: torch.Tensor,
    high: float = 0.99,
    low: float = 0.50,
    quiet_window: int = 5,
) -> dict[str, torch.Tensor]:
    """Flag networks whose complete-domain accuracy jumps late.

    ``domain_acc`` is [P, T] in chronological order. A transition is recorded
    at the first time t with acc >= ``high`` if the previous ``quiet_window``
    logged points were all < ``low``. This is a detector, not a claim that
    grokking occurred.
    """
    if domain_acc.ndim != 2:
        raise ValueError("domain_acc must be [P, T]")
    p, t = domain_acc.shape
    first_high = torch.full((p,), -1, dtype=torch.long)
    crossed = domain_acc >= high
    has = crossed.any(dim=1)
    # argmax on bool returns the first True.
    first_high[has] = crossed.float().argmax(dim=1)[has]

    sudden = torch.zeros(p, dtype=torch.bool)
    for i in range(p):
        t0 = int(first_high[i])
        if t0 < 0:
            continue
        start = max(0, t0 - quiet_window)
        if t0 == 0:
            continue
        if bool((domain_acc[i, start:t0] < low).all()):
            sudden[i] = True

    return {
        "first_exact_log": first_high,
        "sudden_transition": sudden,
        "final_domain_acc": domain_acc[:, -1] if t else torch.zeros(p),
    }
