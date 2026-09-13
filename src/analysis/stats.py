"""Small frequentist helpers. No modeling assumptions beyond binomial sampling."""

from __future__ import annotations

import math


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Return (p_hat, lo, hi) for a binomial proportion."""
    if n <= 0:
        return 0.0, 0.0, 0.0
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    margin = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n) / denom
    return p, max(0.0, center - margin), min(1.0, center + margin)
