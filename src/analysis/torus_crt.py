"""T^m CRT / multi-head torus certificates for Model C (H>=2).

A head is an S^1 factor with image order d = (p-1)/gcd(k, p-1).
Dead and redundant heads are dropped before counting m. Faithful torus
certification requires lcm(d_h)=p-1, a neural-free product NN decoder
exact on all (p-1)^2 pairs, and ablation of each selected head.

Family Q (quotient x C2) remains class2_mod4 / certify_class2_net.
gcd=2 alone is not T^2.

This module is numpy-only (no torch import) so certificates can run
without a GPU stack.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

DEAD_SCALE = 1e-3
R2_LIVE = 0.5
TWO_PI = 2.0 * math.pi


def _lcm(a: int, b: int) -> int:
    a, b = int(a), int(b)
    return a // math.gcd(a, b) * b if a and b else 0


def primitive_root(p: int) -> int:
    if p == 2:
        return 1
    phi = p - 1
    factors: list[int] = []
    n = phi
    d = 2
    while d * d <= n:
        if n % d == 0:
            factors.append(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        factors.append(n)
    for g in range(2, p):
        if all(pow(g, phi // f, p) != 1 for f in factors):
            return g
    raise ValueError(f"no primitive root for {p}")


def discrete_log_table(p: int) -> np.ndarray:
    g = primitive_root(p)
    out = np.full(p, -1, dtype=np.int64)
    x = 1
    for k in range(p - 1):
        out[x] = k
        x = (x * g) % p
    return out


def rot2(phi: float, dtype=np.float64) -> np.ndarray:
    c, s = math.cos(phi), math.sin(phi)
    return np.array([[c, -s], [s, c]], dtype=dtype)


def exact_unit_circle(k: int, dlog: np.ndarray, n: int, conjugate: bool) -> np.ndarray:
    theta = TWO_PI * k * np.array([int(dlog[i + 1]) for i in range(n)], dtype=np.float64) / n
    s = np.sin(theta)
    if conjugate:
        s = -s
    return np.stack([np.cos(theta), s], axis=1)


def apply_scale_rot(unit: np.ndarray, scale: float, phi: float) -> np.ndarray:
    return scale * (unit @ rot2(phi).T)


def effective_winding(k: int | np.ndarray, conjugate: bool | np.ndarray, n: int) -> np.ndarray:
    k = np.atleast_1d(np.asarray(k, dtype=np.int64))
    conj = np.atleast_1d(np.asarray(conjugate, dtype=bool))
    return np.where(conj, np.mod(-k, n), np.mod(k, n)).astype(np.int64)


def fit_dlog_circle_batch(xy: np.ndarray, dlog: np.ndarray, n: int) -> dict[str, np.ndarray]:
    """Vectorized fit. xy [P, n, 2] -> per-net k, phi, scale, conjugate, r2."""
    pop = xy.shape[0]
    y = np.concatenate([xy[:, :, 0], xy[:, :, 1]], axis=1)
    y_var = ((xy - xy.mean(axis=1, keepdims=True)) ** 2).sum(axis=(1, 2))
    y_var = np.maximum(y_var, 1e-18)
    best_r2 = np.full(pop, -1.0)
    best_k = np.zeros(pop, dtype=np.int64)
    best_phi = np.zeros(pop)
    best_scale = np.zeros(pop)
    best_conj = np.zeros(pop, dtype=bool)
    theta0 = TWO_PI * np.array([int(dlog[i + 1]) for i in range(n)], dtype=np.float64) / n
    for k in range(0, n):
        for conj in (False, True):
            c = np.cos(k * theta0)
            s = np.sin(k * theta0)
            if conj:
                s = -s
            B = np.vstack([np.column_stack([c, -s]), np.column_stack([s, c])])
            coef, *_ = np.linalg.lstsq(B, y.T, rcond=None)
            a, bvec = coef[0], coef[1]
            fitted = np.stack(
                [a[:, None] * c - bvec[:, None] * s, bvec[:, None] * c + a[:, None] * s],
                axis=-1,
            )
            r2 = 1.0 - ((xy - fitted) ** 2).sum(axis=(1, 2)) / y_var
            better = r2 > best_r2
            best_r2 = np.where(better, r2, best_r2)
            best_k = np.where(better, k, best_k)
            best_phi = np.where(better, np.arctan2(bvec, a), best_phi)
            best_scale = np.where(better, np.hypot(a, bvec), best_scale)
            best_conj = np.where(better, conj, best_conj)
    return {
        "r2": best_r2,
        "k": best_k,
        "phi": best_phi,
        "scale": best_scale,
        "conjugate": best_conj,
    }


def _as_fit(batch: dict[str, np.ndarray], i: int = 0) -> dict[str, Any]:
    return {
        "r2": float(batch["r2"][i]),
        "k": int(batch["k"][i]),
        "phi": float(batch["phi"][i]),
        "scale": float(batch["scale"][i]),
        "conjugate": bool(batch["conjugate"][i]),
    }


def fit_head(xy_h: np.ndarray, dlog: np.ndarray, n: int) -> dict[str, Any]:
    batch = fit_dlog_circle_batch(xy_h[None, ...], dlog, n)
    fit = _as_fit(batch, 0)
    k_eff = int(effective_winding(fit["k"], fit["conjugate"], n)[0])
    gcd = math.gcd(k_eff, n) if k_eff else n
    fit["k_eff"] = k_eff
    fit["gcd"] = gcd
    fit["d"] = n // gcd
    fit["mean_radius"] = float(np.linalg.norm(xy_h, axis=-1).mean())
    return fit


def is_dead_head(fit: dict[str, Any], xy_h: np.ndarray | None = None) -> bool:
    if fit["scale"] < DEAD_SCALE:
        return True
    if fit.get("mean_radius", fit["scale"]) < DEAD_SCALE:
        return True
    if fit["r2"] < R2_LIVE and fit["scale"] < 0.05:
        return True
    if xy_h is not None and float(np.linalg.norm(xy_h, axis=-1).max()) < DEAD_SCALE:
        return True
    return False


def _winding_key(fit: dict[str, Any], n: int) -> tuple[int, int]:
    k = int(fit["k_eff"]) % n
    g = math.gcd(k, n) if k else n
    if g == n:
        return (0, 0)
    d = n // g
    u = (k // g) % d
    u = min(u, (-u) % d)
    return (g, u)


def dedupe_heads(fits: list[dict[str, Any]], n: int) -> list[int]:
    seen: set[tuple[int, int]] = set()
    out: list[int] = []
    for i, f in enumerate(fits):
        if f.get("dead"):
            continue
        key = _winding_key(f, n)
        if key in seen:
            continue
        seen.add(key)
        out.append(i)
    return out


def select_torus_basis(fits: list[dict[str, Any]], live_idx: list[int], n: int) -> list[int]:
    if not live_idx:
        return []
    ordered = sorted(live_idx, key=lambda i: fits[i]["d"], reverse=True)
    chosen: list[int] = []
    cur = 1
    for i in ordered:
        d = int(fits[i]["d"])
        if d <= 1:
            continue
        new = _lcm(cur, d)
        if new != cur:
            chosen.append(i)
            cur = new
        if cur == n:
            break
    if cur != n:
        return []
    return chosen


def character_table(
    fit: dict[str, Any], dlog: np.ndarray, n: int, *, product_space: bool = False
) -> np.ndarray:
    unit = exact_unit_circle(int(fit["k"]), dlog, n, bool(fit["conjugate"]))
    if product_space:
        return apply_scale_rot(unit, float(fit["scale"]) ** 2, 2.0 * float(fit["phi"]))
    return apply_scale_rot(unit, float(fit["scale"]), float(fit["phi"]))


def complex_mul_table(za: np.ndarray, zb: np.ndarray) -> np.ndarray:
    xa, ya = za[:, 0], za[:, 1]
    xb, yb = zb[:, 0], zb[:, 1]
    real = xa[:, None] * xb[None, :] - ya[:, None] * yb[None, :]
    imag = xa[:, None] * yb[None, :] + ya[:, None] * xb[None, :]
    return np.stack([real, imag], axis=-1)


def torus_product_nn_decode(
    fits: list[dict[str, Any]],
    head_indices: list[int],
    dlog: np.ndarray,
    n: int,
) -> dict[str, Any]:
    if not head_indices:
        return {"exact": False, "acc": 0.0, "n_correct": 0, "n_pairs": n * n}

    emb = [character_table(fits[hi], dlog, n, product_space=False) for hi in head_indices]
    tgt = [character_table(fits[hi], dlog, n, product_space=True) for hi in head_indices]
    prods = [complex_mul_table(e, e) for e in emb]
    prod = np.concatenate(prods, axis=-1)
    target = np.concatenate(tgt, axis=-1)

    diff = prod[:, :, None, :] - target[None, None, :, :]
    dist2 = np.sum(diff * diff, axis=-1)
    pred = dist2.argmin(axis=-1)
    residues = np.arange(1, n + 1)
    true = (residues[:, None] * residues[None, :]) % (n + 1) - 1
    n_ok = int((pred == true).sum())
    n_pairs = n * n
    return {
        "exact": bool(n_ok == n_pairs),
        "acc": n_ok / n_pairs,
        "n_correct": n_ok,
        "n_pairs": n_pairs,
        "pred": pred,
    }


def ablation_breaks(
    fits: list[dict[str, Any]],
    head_indices: list[int],
    dlog: np.ndarray,
    n: int,
) -> dict[str, Any]:
    if not head_indices:
        return {"ok": False, "per_head": []}
    if len(head_indices) == 1:
        dec0 = torus_product_nn_decode(fits, [], dlog, n)
        return {
            "ok": not dec0["exact"],
            "per_head": [{"head": head_indices[0], "exact_after": dec0["exact"]}],
        }

    rows = []
    all_break = True
    for drop in head_indices:
        kept = [h for h in head_indices if h != drop]
        dec = torus_product_nn_decode(fits, kept, dlog, n)
        rows.append({"head": drop, "exact_after": dec["exact"], "acc_after": dec["acc"]})
        if dec["exact"]:
            all_break = False
    return {"ok": bool(all_break), "per_head": rows}


def certify_torus_net(xy: np.ndarray, p: int) -> dict[str, Any]:
    """Certify a multi-head embedding as T^m.

    xy: [n, H, 2] with n = p-1.
    """
    xy = np.asarray(xy, dtype=np.float64)
    if xy.ndim != 3 or xy.shape[0] != p - 1 or xy.shape[2] != 2:
        raise ValueError(f"expected xy [p-1, H, 2], got {xy.shape} for p={p}")
    n, heads, _ = xy.shape
    dlog = discrete_log_table(p)

    out: dict[str, Any] = {
        "certified": False,
        "m": 0,
        "heads": heads,
        "d_h": [],
        "k_h": [],
        "k_eff": [],
        "lcm": 1,
        "decoder_exact": False,
        "decoder_acc": 0.0,
        "ablation_ok": False,
        "break_at": None,
        "label": "EXACT-UNCLASSIFIED",
        "selected_heads": [],
        "live_heads": [],
        "torus_m": 0,
    }

    fits: list[dict[str, Any]] = []
    for h in range(heads):
        f = fit_head(xy[:, h, :], dlog, n)
        f["dead"] = is_dead_head(f, xy[:, h, :])
        fits.append(f)
        out["d_h"].append(int(f["d"]))
        out["k_h"].append(int(f["k"]))
        out["k_eff"].append(int(f["k_eff"]))
    out["fits"] = [{k: v for k, v in f.items() if k != "fitted"} for f in fits]

    live = dedupe_heads(fits, n)
    out["live_heads"] = list(live)
    if not live:
        out["break_at"] = "no_live_heads"
        return out

    basis = select_torus_basis(fits, live, n)
    out["selected_heads"] = list(basis)
    if not basis:
        cur = 1
        for i in live:
            cur = _lcm(cur, int(fits[i]["d"]))
        out["lcm"] = int(cur)
        out["m"] = len(live)
        out["break_at"] = "lcm_not_n"
        return out

    m = len(basis)
    cur = 1
    for i in basis:
        cur = _lcm(cur, int(fits[i]["d"]))
    out["m"] = m
    out["lcm"] = int(cur)
    out["torus_m"] = m

    dec = torus_product_nn_decode(fits, basis, dlog, n)
    out["decoder_exact"] = bool(dec["exact"])
    out["decoder_acc"] = float(dec["acc"])
    if not dec["exact"]:
        out["break_at"] = "decoder_not_exact"
        return out

    abl = ablation_breaks(fits, basis, dlog, n)
    out["ablation_ok"] = bool(abl["ok"])
    out["ablation"] = abl["per_head"]
    if not abl["ok"]:
        out["break_at"] = "ablation_still_exact"
        return out

    out["certified"] = True
    out["break_at"] = None
    out["label"] = f"TORUS_T{m}"
    if m == 1 and int(fits[basis[0]]["gcd"]) == 1:
        out["label"] = "FAMILY_F"
    return out


def try_family_q_on_head(xy_h: np.ndarray, fit: dict[str, Any], p: int) -> dict[str, Any]:
    """Run locked Class-2 / Family Q certificate on one head only."""
    from src.analysis.class2_mod4 import certify_class2_net

    return certify_class2_net(xy_h, int(fit["k_eff"]), float(fit["phi"]), p, None)
