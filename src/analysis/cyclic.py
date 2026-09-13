"""Per-prime cyclic-group metadata and Stage A predictions.

Predictions P1–P7 are recorded before any Stage A training.
"""

from __future__ import annotations

from src.analysis.fourier import primitive_root
from src.analysis.group_law import euler_phi, units_mod


STAGE_A_PRIMES = (7, 11, 13, 17)
STAGE_B_PRIMES = (19, 23, 29, 31)

PREDICTIONS = {
    "P1": "Exact generalizers use gcd(k, p-1)=1.",
    "P2": "Held-row centered homomorphism error is about as small as train-pair error.",
    "P3": "Exact aligned roots-of-unity + frozen readout preserves exactness for H=1 generalizers.",
    "P4": "Scrambled tables will not show faithful-winding + symbolic substitution (Stage C, not run here).",
    "P5": "Latent embedding dimension stays 2 for all tested cyclic groups.",
    "P6": "P(exact) may vary strongly with p; that does not by itself falsify the representation.",
    "P7": "Some faithful windings will be discovered more often than others.",
}


def prime_card(p: int) -> dict:
    n = p - 1
    units = units_mod(n)
    return {
        "p": p,
        "n": n,
        "primitive_root": int(primitive_root(p)),
        "phi": euler_phi(n),
        "units": units,
        "n_units": len(units),
        "frac_k_units": len(units) / n,
        "n_domain": n * n,
        "n_train_held_row": (n - 1) * n,
        "n_test_held_row": n,
        "params_h1": 5 * n,
        "held_residue": n,
        "p_minus_1_factorization": _factor(n),
    }


def _factor(n: int) -> list[int]:
    factors = []
    x = n
    d = 2
    while d * d <= x:
        while x % d == 0:
            factors.append(d)
            x //= d
        d += 1
    if x > 1:
        factors.append(x)
    return factors


def all_cards(primes=STAGE_A_PRIMES) -> list[dict]:
    return [prime_card(p) for p in primes]
