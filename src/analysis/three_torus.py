"""3-torus certificate for Model C H=3 on F_31*: C_2 × C_3 × C_5 via CRT.

A winding k of C_30 has image order 30/gcd(k,30). Orders 2,3,5 are gcd
15,10,6. R² assigns k; the gate is the neural-free CRT program.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from src.analysis.fourier import discrete_log_table, primitive_root
from src.analysis.group_law import TWO_PI, exact_unit_circle, wrap
from src.analysis.level4_h1 import effective_winding


P = 31
N = 30
HEADS = 3
ORDER_GCD = {2: 15, 3: 10, 5: 6}
GCD_ORDER = {15: 2, 10: 3, 6: 5}
TORUS_GCDS = frozenset({15, 10, 6})


def image_order(k: int, n: int = N) -> int:
    k = int(k) % n
    if k == 0:
        return 1
    return n // math.gcd(k, n)


def t_unit(k: int, d: int, n: int = N) -> int:
    """k = (n/d)*t with gcd(t,d)=1 for an order-d character."""
    step = n // d
    if int(k) % step != 0:
        raise ValueError(f"k={k} is not an order-{d} winding")
    t = (int(k) // step) % d
    if math.gcd(t, d) != 1:
        raise ValueError(f"k={k} gives t={t} not coprime to {d}")
    return t


def coord_mod_d(xy: np.ndarray, k_eff: int, phi: float, d: int, n: int = N) -> np.ndarray:
    """j mod d from nearest of d cyclotomic slots."""
    theta = np.arctan2(xy[:, 1], xy[:, 0])
    ang = wrap(theta - float(phi))
    slot = np.mod(np.round(ang * d / TWO_PI), d).astype(np.int64)
    t = t_unit(int(k_eff) % n, d, n)
    inv = pow(int(t), -1, d)
    return (slot * inv) % d


def crt_235(j2: int, j3: int, j5: int) -> int:
    for j in range(N):
        if j % 2 == j2 % 2 and j % 3 == j3 % 3 and j % 5 == j5 % 5:
            return j
    raise RuntimeError("CRT failed")


def assign_heads(gcds: list[int]) -> dict[int, int] | None:
    """Map order -> head index if gcds are a permutation of {15,10,6}."""
    if frozenset(gcds) != TORUS_GCDS or len(gcds) != 3:
        return None
    return {GCD_ORDER[g]: i for i, g in enumerate(gcds)}


def crt_decode_table(
    coords: dict[int, np.ndarray],
    g: int,
    p: int = P,
) -> np.ndarray:
    """Neural-free labels 0..p-2 for every (a,b). coords[d] is j mod d on residues 1..n."""
    n = p - 1
    out = np.empty((n, n), dtype=np.int64)
    for ia in range(n):
        for ib in range(n):
            s2 = (int(coords[2][ia]) + int(coords[2][ib])) % 2
            s3 = (int(coords[3][ia]) + int(coords[3][ib])) % 3
            s5 = (int(coords[5][ia]) + int(coords[5][ib])) % 5
            j = crt_235(s2, s3, s5)
            out[ia, ib] = pow(int(g), j, p) - 1
    return out


def truth_table(p: int = P) -> np.ndarray:
    n = p - 1
    a = np.arange(1, p)
    return ((a[:, None] * a[None, :]) % p) - 1


def certify_net(
    E: np.ndarray,
    k_eff: np.ndarray,
    phi: np.ndarray,
    *,
    p: int = P,
    g: int | None = None,
) -> dict[str, Any]:
    """E [n, H, 2], k_eff/phi [H]."""
    n = p - 1
    g = int(g if g is not None else primitive_root(p))
    gcds = [math.gcd(int(k) % n, n) if int(k) % n else n for k in k_eff]
    mapping = assign_heads(gcds)
    truth = truth_table(p)
    out: dict[str, Any] = {
        "gcds": gcds,
        "orders": [image_order(int(k), n) for k in k_eff],
        "k_eff": [int(k) % n for k in k_eff],
        "has_faithful_head": any(g == 1 for g in gcds),
        "candidate_triple": mapping is not None,
        "certified": False,
        "break_at": None,
        "crt_n_correct": 0,
        "crt_n": n * n,
    }
    if mapping is None:
        out["break_at"] = "not_gcd_triple_15_10_6"
        return out
    try:
        coords = {}
        for d, h in mapping.items():
            coords[d] = coord_mod_d(E[:, h, :], int(k_eff[h]), float(phi[h]), d, n)
    except ValueError as e:
        out["break_at"] = f"coord_extract:{e}"
        return out
    pred = crt_decode_table(coords, g, p)
    n_ok = int((pred == truth).sum())
    out["crt_n_correct"] = n_ok
    if n_ok != n * n:
        out["break_at"] = "crt_decoder_not_exact"
        return out
    for d in (2, 3, 5):
        killed = {k: v.copy() for k, v in coords.items()}
        rng = np.random.default_rng(31 * 1000 + d)
        killed[d] = rng.integers(0, d, size=n)
        pred_k = crt_decode_table(killed, g, p)
        if int((pred_k == truth).sum()) == n * n:
            out["break_at"] = f"drop_factor_{d}_still_exact"
            return out
    snapped = np.empty_like(E)
    dlog = discrete_log_table(p)
    for h in range(E.shape[1]):
        rho = float(np.linalg.norm(E[:, h, :], axis=1).mean())
        unit = exact_unit_circle(int(k_eff[h]) % n, dlog, n, False)
        # k_eff already includes conjugate; exact_unit_circle(k_eff, conj=False)
        c, s = math.cos(float(phi[h])), math.sin(float(phi[h]))
        rot = np.array([[c, -s], [s, c]])
        snapped[:, h, :] = rho * (unit @ rot.T)
    try:
        snap_coords = {
            d: coord_mod_d(snapped[:, h, :], int(k_eff[h]), float(phi[h]), d, n)
            for d, h in mapping.items()
        }
    except ValueError as e:
        out["break_at"] = f"snap_extract:{e}"
        return out
    snap_pred = crt_decode_table(snap_coords, g, p)
    if int((snap_pred == truth).sum()) != n * n:
        out["break_at"] = "snap_not_exact"
        return out
    out["certified"] = True
    out["head_of_order"] = mapping
    return out


def synthetic_embeddings(p: int = P) -> np.ndarray:
    dlog = discrete_log_table(p)
    n = p - 1
    E = np.zeros((n, HEADS, 2), dtype=np.float64)
    for h, k in enumerate((15, 10, 6)):
        E[:, h, :] = exact_unit_circle(k, dlog, n, False)
    return E


def decoder_sanity() -> dict[str, Any]:
    E = synthetic_embeddings()
    k_eff = np.array([15, 10, 6])
    phi = np.zeros(3)
    cert = certify_net(E, k_eff, phi)
    return {"passed": bool(cert["certified"]), "cert": cert}
