"""Valuation-min certificate for Model T on gcd tables.

gcd(a,b) = product_p p^{min(v_p(a), v_p(b))}. Heads are assigned to the
locked primes by affine fit; the gate is the neural-free min decoder.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.tasks.gcd import DESTROY_SEED, gcd_destroyed, gcd_meet, gcd_table, primes_upto

N = 12
PRIMES = primes_upto(N)
HEADS = len(PRIMES)


def valuations(n: int = N, primes: tuple[int, ...] = PRIMES) -> np.ndarray:
    v = np.zeros((n, len(primes)), dtype=np.float64)
    for i, a in enumerate(range(1, n + 1)):
        for j, p in enumerate(primes):
            x = a
            e = 0
            while x % p == 0:
                x //= p
                e += 1
            v[i, j] = e
    return v


def affine_fit(v: np.ndarray, e: np.ndarray) -> tuple[float, float, float]:
    v = np.asarray(v, dtype=np.float64)
    e = np.asarray(e, dtype=np.float64)
    if float(np.std(v)) < 1e-12:
        return 0.0, float(e.mean()), 0.0
    a = np.stack([v, np.ones_like(v)], axis=1)
    coef, *_ = np.linalg.lstsq(a, e, rcond=None)
    pred = a @ coef
    ss_res = float(((e - pred) ** 2).sum())
    ss_tot = float(((e - e.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return float(coef[0]), float(coef[1]), float(r2)


def assign_heads(E: np.ndarray, v: np.ndarray) -> dict[str, Any]:
    """Map each prime index to a unique head by maximizing affine R²."""
    n, h = E.shape
    n_p = v.shape[1]
    if h != n_p:
        return {"ok": False, "reason": f"H={h} != n_primes={n_p}"}
    r2 = np.zeros((h, n_p), dtype=np.float64)
    alpha = np.zeros((h, n_p), dtype=np.float64)
    beta = np.zeros((h, n_p), dtype=np.float64)
    for i in range(h):
        for j in range(n_p):
            alpha[i, j], beta[i, j], r2[i, j] = affine_fit(v[:, j], E[:, i])
    rows, cols = linear_sum_assignment(-r2)
    head_of_prime = {int(j): int(i) for i, j in zip(rows, cols)}
    scales = []
    intercepts = []
    r2s = []
    for j in range(n_p):
        i = head_of_prime[j]
        scales.append(float(alpha[i, j]))
        intercepts.append(float(beta[i, j]))
        r2s.append(float(r2[i, j]))
    if any(s <= 1e-8 for s in scales):
        return {
            "ok": False,
            "reason": "nonpositive_scale",
            "head_of_prime": head_of_prime,
            "alpha": scales,
            "beta": intercepts,
            "r2": r2s,
        }
    return {
        "ok": True,
        "head_of_prime": head_of_prime,
        "alpha": scales,
        "beta": intercepts,
        "r2": r2s,
    }


def snap_embeddings(E: np.ndarray, v: np.ndarray, assignment: dict[str, Any]) -> np.ndarray:
    snapped = np.empty_like(E, dtype=np.float64)
    for j, h in assignment["head_of_prime"].items():
        snapped[:, h] = assignment["alpha"][j] * v[:, j] + assignment["beta"][j]
    return snapped


def reconstruct(exps: np.ndarray, primes: tuple[int, ...] = PRIMES) -> np.ndarray:
    out = np.ones(exps.shape[:-1], dtype=np.int64)
    for j, p in enumerate(primes):
        e = np.clip(np.rint(exps[..., j]), 0, 16).astype(np.int64)
        out *= np.power(p, e, dtype=np.int64)
    return out


def decode_table(
    snapped: np.ndarray,
    assignment: dict[str, Any],
    primes: tuple[int, ...] = PRIMES,
    op: str = "min",
) -> np.ndarray:
    n, _ = snapped.shape
    n_p = len(primes)
    exps = np.empty((n, n, n_p), dtype=np.float64)
    for j in range(n_p):
        h = assignment["head_of_prime"][j]
        ea = snapped[:, h][:, None]
        eb = snapped[:, h][None, :]
        if op == "min":
            m = np.minimum(ea, eb)
        elif op == "sum":
            m = ea + eb
        else:
            raise ValueError(op)
        exps[:, :, j] = (m - assignment["beta"][j]) / assignment["alpha"][j]
    return reconstruct(exps, primes) - 1


def truth_labels(table: np.ndarray) -> np.ndarray:
    return table.astype(np.int64) - 1


def certify_net(
    E: np.ndarray,
    table: np.ndarray,
    *,
    primes: tuple[int, ...] = PRIMES,
    scramble_seed: int = 0,
) -> dict[str, Any]:
    n = table.shape[0]
    v = valuations(n, primes)
    truth = truth_labels(table)
    out: dict[str, Any] = {
        "certified": False,
        "break_at": None,
        "min_n_correct": 0,
        "min_n": n * n,
        "sum_n_correct": 0,
    }
    assignment = assign_heads(E, v)
    out["assignment"] = {
        k: assignment[k]
        for k in ("ok", "reason", "head_of_prime", "alpha", "beta", "r2")
        if k in assignment
    }
    if not assignment.get("ok"):
        out["break_at"] = assignment.get("reason", "assignment_failed")
        return out
    snapped = snap_embeddings(E, v, assignment)
    pred = decode_table(snapped, assignment, primes, op="min")
    n_ok = int((pred == truth).sum())
    out["min_n_correct"] = n_ok
    out["r2"] = assignment["r2"]
    out["head_of_prime"] = assignment["head_of_prime"]
    if n_ok != n * n:
        out["break_at"] = "min_decoder_not_exact"
        return out
    pred_sum = decode_table(snapped, assignment, primes, op="sum")
    out["sum_n_correct"] = int((pred_sum == truth).sum())
    if out["sum_n_correct"] == n * n:
        out["break_at"] = "sum_decoder_still_exact"
        return out
    for j, p in enumerate(primes):
        h = assignment["head_of_prime"][j]
        killed = snapped.copy()
        rng = np.random.default_rng(scramble_seed * 1009 + p)
        killed[:, h] = rng.permutation(killed[:, h])
        pred_k = decode_table(killed, assignment, primes, op="min")
        if int((pred_k == truth).sum()) == n * n:
            out["break_at"] = f"drop_prime_{p}_still_exact"
            return out
    out["certified"] = True
    return out


def synthetic_embeddings(n: int = N, primes: tuple[int, ...] = PRIMES) -> np.ndarray:
    return valuations(n, primes).copy()


def decoder_sanity() -> dict[str, Any]:
    E = synthetic_embeddings()
    meet = gcd_table(N)
    destroyed = gcd_destroyed(N, DESTROY_SEED).table
    cert_meet = certify_net(E, meet, scramble_seed=1)
    cert_destroyed = certify_net(E, destroyed, scramble_seed=1)
    return {
        "passed": bool(cert_meet["certified"]) and (not bool(cert_destroyed["certified"])),
        "meet": cert_meet,
        "destroyed": cert_destroyed,
        "destroyed_equals_gcd": bool(np.array_equal(destroyed, meet)),
    }
