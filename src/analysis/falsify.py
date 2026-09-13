"""Falsifiers for the discrete-log / character story.

Measurements only. A family "survives" a relabeling when its R² stays
high on the transformed embedding. Survival under every relabeling is
evidence of flexibility, not of group structure.
"""

from __future__ import annotations

import numpy as np
import torch

from src.analysis.fourier import _best_dlog_r2, _best_sinusoid_r2, _r2, discrete_log_table


def additive_perm(p: int, k: int) -> np.ndarray:
    return np.array([(a + k) % p for a in range(p)], dtype=np.int64)


def multiplicative_perm(p: int, c: int) -> np.ndarray:
    """0 stays 0; nonzero residues multiply by c in F_p."""
    if c % p == 0:
        raise ValueError("multiplicative constant must be nonzero")
    return np.array([0 if a == 0 else (c * a) % p for a in range(p)], dtype=np.int64)


def random_nonzero_perm(p: int, rng: np.random.Generator) -> np.ndarray:
    perm = np.arange(p, dtype=np.int64)
    perm[1:] = rng.permutation(np.arange(1, p, dtype=np.int64))
    return perm


def apply_perm(y: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """New vector at residue a is the old value at perm[a]."""
    return np.asarray(y, dtype=np.float64)[perm]


def family_r2(y: np.ndarray, p: int, dlog: np.ndarray | None = None) -> dict[str, float]:
    y = np.asarray(y, dtype=np.float64)
    if dlog is None:
        dlog = discrete_log_table(p)
    f_r2, f_k = _best_sinusoid_r2(y)
    d_r2, d_k = _best_dlog_r2(y, dlog)
    return {"fourier_r2": f_r2, "fourier_k": int(f_k), "dlog_r2": d_r2, "dlog_k": int(d_k)}


def orbit_r2(
    y: np.ndarray,
    p: int,
    rng: np.random.Generator,
    n_random: int = 12,
) -> dict:
    """R² of one embedding under additive, multiplicative, and random orbits."""
    dlog = discrete_log_table(p)
    base = family_r2(y, p, dlog)
    add, mul, rnd = [], [], []
    for k in range(1, p):
        add.append(family_r2(apply_perm(y, additive_perm(p, k)), p, dlog))
        mul.append(family_r2(apply_perm(y, multiplicative_perm(p, k)), p, dlog))
    for _ in range(n_random):
        rnd.append(family_r2(apply_perm(y, random_nonzero_perm(p, rng)), p, dlog))

    def _pack(rows: list[dict]) -> dict:
        d = np.array([r["dlog_r2"] for r in rows])
        f = np.array([r["fourier_r2"] for r in rows])
        return {
            "mean_dlog": float(d.mean()),
            "min_dlog": float(d.min()),
            "max_dlog": float(d.max()),
            "mean_fourier": float(f.mean()),
            "frac_dlog_ge_085": float((d >= 0.85).mean()),
            "frac_fourier_ge_085": float((f >= 0.85).mean()),
        }

    return {"base": base, "additive": _pack(add), "multiplicative": _pack(mul), "random": _pack(rnd)}


def grouped_r2(values: np.ndarray, keys: np.ndarray) -> float:
    """R² of predicting each entry by the mean of its key-class."""
    values = np.asarray(values, dtype=np.float64).ravel()
    keys = np.asarray(keys).ravel()
    pred = np.empty_like(values)
    for k in np.unique(keys):
        m = keys == k
        pred[m] = values[m].mean()
    return _r2(values, pred)


def hidden_factor_r2(h_ab: np.ndarray, p: int) -> dict[str, float]:
    """How much of a hidden map is a function of one discrete invariant."""
    h = np.asarray(h_ab, dtype=np.float64)
    assert h.shape == (p, p)
    aa, bb = np.indices((p, p))
    dlog = discrete_log_table(p)
    prod = (aa * bb) % p
    add = (aa + bb) % p
    # Nonzero pairs: dlog(a)+dlog(b) and dlog(a)-dlog(b). Zeros get sentinels.
    dsum = np.full((p, p), -99, dtype=np.int64)
    ddiff = np.full((p, p), -99, dtype=np.int64)
    nz = (aa > 0) & (bb > 0)
    dsum[nz] = (dlog[aa[nz]] + dlog[bb[nz]]) % (p - 1)
    ddiff[nz] = (dlog[aa[nz]] - dlog[bb[nz]]) % (p - 1)
    dsum[aa == 0] = -1
    dsum[(aa > 0) & (bb == 0)] = -2
    ddiff[aa == 0] = -1
    ddiff[(aa > 0) & (bb == 0)] = -2
    zero_pat = (aa == 0).astype(np.int64) + 2 * (bb == 0).astype(np.int64)
    return {
        "product": grouped_r2(h, prod),
        "sum": grouped_r2(h, add),
        "a": grouped_r2(h, aa),
        "b": grouped_r2(h, bb),
        "dlog_sum": grouped_r2(h, dsum),
        "dlog_diff": grouped_r2(h, ddiff),
        "zero_pattern": grouped_r2(h, zero_pat),
    }


@torch.no_grad()
def collect_embeddings(u: torch.Tensor, v: torch.Tensor) -> list[np.ndarray]:
    """Flatten population × unit × side into a list of [p] vectors."""
    out = []
    u_np = u.detach().cpu().numpy()
    v_np = v.detach().cpu().numpy()
    for i in range(u_np.shape[0]):
        for h in range(u_np.shape[1]):
            out.append(u_np[i, h])
            out.append(v_np[i, h])
    return out
