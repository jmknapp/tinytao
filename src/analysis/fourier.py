"""Fits of first-layer residue embeddings to simple 1-D families.

Every hidden unit of this architecture is exactly
    relu(u[a] + v[b] + c)
so characterizing u and v is characterizing the unit. We report R^2 of
the best fit in each family. High R^2 is a measurement, not a proof that
the network "uses" that representation.
"""

from __future__ import annotations

import math

import numpy as np
import torch


def primitive_root(p: int) -> int:
    """Smallest primitive root modulo prime p."""
    if p == 2:
        return 1
    phi = p - 1
    # factor phi
    factors = []
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
    """dlog[a] in 0..p-2 for a≠0; -1 for 0."""
    g = primitive_root(p)
    out = np.full(p, -1, dtype=np.int64)
    x = 1
    for k in range(p - 1):
        out[x] = k
        x = (x * g) % p
    return out


def _r2(y: np.ndarray, yhat: np.ndarray) -> float:
    y = y.astype(np.float64)
    yhat = yhat.astype(np.float64)
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    if ss_tot <= 1e-18:
        return 1.0 if ss_res <= 1e-18 else 0.0
    return max(0.0, 1.0 - ss_res / ss_tot)


def _fit_sinusoid(y: np.ndarray, t: np.ndarray, period: int) -> tuple[np.ndarray, float, int]:
    """Best constant + A cos(2πkt/period) + B sin(...). Returns yhat, R², k."""
    y = np.asarray(y, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    ones = np.ones_like(y)
    best_r2, best_k, best_hat = -1.0, 0, y.copy()
    for k in range(1, period // 2 + 1):
        c = np.cos(2 * math.pi * k * t / period)
        s = np.sin(2 * math.pi * k * t / period)
        X = np.stack([ones, c, s], axis=1)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        yhat = X @ coef
        r2 = _r2(y, yhat)
        if r2 > best_r2:
            best_r2, best_k, best_hat = r2, k, yhat
    return best_hat, best_r2, best_k


def _best_sinusoid_r2(y: np.ndarray) -> tuple[float, int]:
    """Best constant + A cos(2πkt/n) + B sin(...) over k=1..floor((n-1)/2)."""
    n = y.size
    _yhat, r2, k = _fit_sinusoid(y, np.arange(n, dtype=np.float64), n)
    return r2, k


def reconstruct_residue_sinusoid(y: np.ndarray) -> tuple[np.ndarray, float, int]:
    """Replace y with its best residue-axis sinusoid (all p points)."""
    y = np.asarray(y, dtype=np.float64)
    return _fit_sinusoid(y, np.arange(y.size, dtype=np.float64), y.size)


def reconstruct_dlog_sinusoid(y: np.ndarray, dlog: np.ndarray) -> tuple[np.ndarray, float, int]:
    """Replace nonzero residues with the best dlog sinusoid; keep y[0]."""
    y = np.asarray(y, dtype=np.float64)
    dlog = np.asarray(dlog)
    out = y.copy()
    nz = dlog >= 0
    if nz.sum() < 4:
        return out, 0.0, 0
    period = int(dlog.max() + 1)
    yhat, r2, k = _fit_sinusoid(y[nz], dlog[nz].astype(np.float64), period)
    out[nz] = yhat
    return out, r2, k


def _best_onehot_r2(y: np.ndarray) -> tuple[float, int]:
    """Best constant + spike at a single residue."""
    n = y.size
    ones = np.ones(n)
    best_r2, best_i = -1.0, 0
    for i in range(n):
        spike = np.zeros(n)
        spike[i] = 1.0
        X = np.stack([ones, spike], axis=1)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        r2 = _r2(y, X @ coef)
        if r2 > best_r2:
            best_r2, best_i = r2, i
    return best_r2, best_i


def _best_dlog_r2(y: np.ndarray, dlog: np.ndarray) -> tuple[float, int]:
    """Best sinusoid in discrete-log, evaluated on a≠0 only."""
    _yhat, r2, k = reconstruct_dlog_sinusoid(y, dlog)
    return r2, k


def describe_embedding(vec: torch.Tensor | np.ndarray, p: int, dlog: np.ndarray | None = None) -> dict:
    """R^2 of one 1-D embedding against sinusoid / one-hot / dlog families."""
    y = np.asarray(vec if not torch.is_tensor(vec) else vec.detach().cpu().numpy(), dtype=np.float64)
    if dlog is None:
        dlog = discrete_log_table(p)
    spec = np.abs(np.fft.rfft(y))
    spec = spec / (spec.sum() + 1e-12)
    f_r2, f_k = _best_sinusoid_r2(y)
    o_r2, o_i = _best_onehot_r2(y)
    d_r2, d_k = _best_dlog_r2(y, dlog)
    return {
        "fourier_r2": f_r2,
        "fourier_k": f_k,
        "onehot_r2": o_r2,
        "onehot_residue": o_i,
        "dlog_r2": d_r2,
        "dlog_k": d_k,
        "spectrum": spec.tolist(),
        "dominant_bin": int(np.argmax(spec)),
    }


def describe_network_embeddings(u: torch.Tensor, v: torch.Tensor, p: int) -> list[list[dict]]:
    """Per-network, per-unit descriptors of (u, v). u,v are [P, H, p]."""
    dlog = discrete_log_table(p)
    out = []
    for i in range(u.shape[0]):
        units = []
        for h in range(u.shape[1]):
            ua = u[i, h].detach().cpu().numpy()
            va = v[i, h].detach().cpu().numpy()
            if ua.std() < 1e-12 or va.std() < 1e-12:
                uv_corr = 0.0
            else:
                uv_corr = float(np.corrcoef(ua, va)[0, 1])
            units.append(
                {
                    "u": describe_embedding(ua, p, dlog),
                    "v": describe_embedding(va, p, dlog),
                    "uv_corr": uv_corr,
                }
            )
        out.append(units)
    return out


def family_votes(units: list[dict], thresh: float = 0.85, gap: float = 0.15) -> dict:
    """Count sides whose best family R² clears ``thresh`` and beats the next by ``gap``."""
    counts = {"fourier": 0, "onehot": 0, "dlog": 0, "ambiguous": 0, "weak": 0, "n": 0}
    for side in ("u", "v"):
        for u in units:
            r = u[side]
            scores = {
                "fourier": r["fourier_r2"],
                "onehot": r["onehot_r2"],
                "dlog": r["dlog_r2"],
            }
            winner, best = max(scores.items(), key=lambda kv: kv[1])
            second = sorted(scores.values(), reverse=True)[1]
            counts["n"] += 1
            if best < thresh:
                counts["weak"] += 1
            elif best - second < gap:
                counts["ambiguous"] += 1
            else:
                counts[winner] += 1
    return counts
