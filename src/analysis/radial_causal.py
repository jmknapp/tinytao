"""Causal interventions for p=11 non-unit exact H=1 solvers.

Predictions live in stage_a5_hypotheses.PREREGISTRATION and must be written
to disk before this module evaluates a modified network.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.analysis.fourier import discrete_log_table
from src.analysis.group_law import TWO_PI
from src.analysis.radial_quotient import fiber_id_from_k, fiber_members, product_hidden


def residue_grid(p: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = p - 1
    a = np.arange(1, p)
    aa, bb = np.meshgrid(a, a, indexing="ij")
    target = (aa * bb) % p
    return aa, bb, target


def dlog_vals(p: int) -> np.ndarray:
    dlog = discrete_log_table(p)
    return np.array([int(dlog[i]) for i in range(1, p)], dtype=np.int64)


def polar_parts(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r = np.linalg.norm(xy, axis=-1)
    th = np.arctan2(xy[:, 1], xy[:, 0])
    return r, th


def from_polar(r: np.ndarray, th: np.ndarray) -> np.ndarray:
    return np.stack([r * np.cos(th), r * np.sin(th)], axis=-1)


def classify_table(xy: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return residues 1..p-1. W [2, C], b [C]."""
    h, _, _ = product_hidden(xy)
    W = np.asarray(W, dtype=np.float64)
    if W.shape[0] != 2:
        W = W.T
    logits = h @ W + np.asarray(b, dtype=np.float64).reshape(1, 1, -1)
    return logits.argmax(axis=-1) + 1


def pair_accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    return float((pred == target).mean())


def is_exact(pred: np.ndarray, target: np.ndarray) -> bool:
    return bool((pred == target).all())


def split_acc(pred: np.ndarray, target: np.ndarray, train_mask: np.ndarray) -> dict[str, float]:
    return {
        "domain": pair_accuracy(pred, target),
        "train": pair_accuracy(pred[train_mask], target[train_mask]),
        "held": pair_accuracy(pred[~train_mask], target[~train_mask]),
        "exact": is_exact(pred, target),
    }


def parity_split_acc(pred: np.ndarray, target: np.ndarray, dlog_v: np.ndarray) -> dict[str, float]:
    tgt_par = dlog_v[target - 1] % 2
    even = tgt_par == 0
    odd = ~even
    return {
        "even_product_parity": pair_accuracy(pred[even], target[even]),
        "odd_product_parity": pair_accuracy(pred[odd], target[odd]),
        "n_even": int(even.sum()),
        "n_odd": int(odd.sum()),
    }


def rho_parity(r: np.ndarray, s: np.ndarray) -> tuple[float, float]:
    return float(r[s == 0].mean()), float(r[s == 1].mean())


def apply_two_level_radii(xy: np.ndarray, s: np.ndarray, rho_even: float, rho_odd: float) -> np.ndarray:
    _, th = polar_parts(xy)
    r = np.where(s == 0, rho_even, rho_odd).astype(np.float64)
    return from_polar(r, th)


def apply_global_level_swap(xy: np.ndarray, s: np.ndarray) -> np.ndarray:
    r, th = polar_parts(xy)
    rho_e, rho_o = rho_parity(r, s)
    return apply_two_level_radii(xy, s, rho_o, rho_e)


def swap_fiber_radii(xy: np.ndarray, u: int, p: int) -> np.ndarray:
    r, th = polar_parts(xy)
    iu, iv = u - 1, (p - u) - 1
    r = r.copy()
    r[iu], r[iv] = r[iv], r[iu]
    return from_polar(r, th)


def apply_exact_phases(xy: np.ndarray, k: int, phi: float, dlog_v: np.ndarray, n: int) -> np.ndarray:
    r, _ = polar_parts(xy)
    th = phi + TWO_PI * int(k) * dlog_v / n
    return from_polar(r, th)


def apply_unit_norm(xy: np.ndarray) -> np.ndarray:
    r, th = polar_parts(xy)
    return from_polar(np.ones_like(r), th)


def n_in_fiber(aa: np.ndarray, bb: np.ndarray, u: int, p: int) -> np.ndarray:
    F = {u, p - u}
    a_in = np.isin(aa, list(F))
    b_in = np.isin(bb, list(F))
    return a_in.astype(np.int64) + b_in.astype(np.int64)


def predict_single_fiber_labels(orig: np.ndarray, n_in: np.ndarray, p: int) -> np.ndarray:
    """Preregistered label permutation."""
    out = orig.copy()
    one = n_in == 1
    out[one] = (p - orig[one]) % p
    return out


def change_metrics(pred_changed: np.ndarray, actual_changed: np.ndarray) -> dict[str, float]:
    pc = pred_changed.astype(bool)
    ac = actual_changed.astype(bool)
    tp = int((pc & ac).sum())
    fp = int((~pc & ac).sum())
    fn = int((pc & ~ac).sum())
    tn = int((~pc & ~ac).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "n_pred": int(pc.sum()),
        "n_actual": int(ac.sum()),
    }


def predict_subset_labels(orig: np.ndarray, aa: np.ndarray, bb: np.ndarray, swapped: set[int], fid: np.ndarray, p: int) -> np.ndarray:
    """Flip to -c iff exactly one of fiber(a), fiber(b) is in swapped."""
    fa = np.array([fid[a - 1] in swapped for a in aa.reshape(-1)], dtype=bool).reshape(aa.shape)
    fb = np.array([fid[b - 1] in swapped for b in bb.reshape(-1)], dtype=bool).reshape(bb.shape)
    flip = fa ^ fb
    out = orig.copy()
    out[flip] = (p - orig[flip]) % p
    return out, flip


def symbolic_decode(
    xy: np.ndarray,
    k: int,
    phi: float,
    dlog_v: np.ndarray,
    p: int,
    rho_even: float,
    rho_odd: float,
) -> np.ndarray:
    """Phase -> {c,-c}; mixed vs same shell -> dlog parity of ab."""
    n = p - 1
    q = n // 2
    h, pr, pth = product_hidden(xy)
    # product phase vs 2 phi + 2π k f / n, f = 0..q-1
    reps = np.arange(q, dtype=np.float64)
    ideal = 2.0 * phi + TWO_PI * int(k) * reps / n
    delta = np.arctan2(np.sin(pth[..., None] - ideal), np.cos(pth[..., None] - ideal))
    prod_fiber = np.abs(delta).argmin(axis=-1)
    shells = np.array([rho_even**2, rho_even * rho_odd, rho_odd**2], dtype=np.float64)
    nearest = np.abs(pr[..., None] - shells).argmin(axis=-1)
    mixed = nearest == 1
    # even product parity if not mixed
    want_even = ~mixed
    # residues in fiber f: dlog ≡ f (mod q)
    pred = np.empty(pr.shape, dtype=np.int64)
    for f in range(q):
        members = [a for a in range(1, p) if int(dlog_v[a - 1] % q) == f]
        even_m = [a for a in members if int(dlog_v[a - 1] % 2) == 0]
        odd_m = [a for a in members if int(dlog_v[a - 1] % 2) == 1]
        even_a = even_m[0] if even_m else members[0]
        odd_a = odd_m[0] if odd_m else members[-1]
        mask = prod_fiber == f
        pred[mask & want_even] = even_a
        pred[mask & ~want_even] = odd_a
    return pred


def permute_radii(xy: np.ndarray, perm: np.ndarray) -> np.ndarray:
    r, th = polar_parts(xy)
    return from_polar(r[perm], th)


def mask_metrics(pred: np.ndarray, actual: np.ndarray, train_mask: np.ndarray, dlog_v: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    out = split_acc(pred, target, train_mask)
    out.update(parity_split_acc(pred, target, dlog_v))
    return out
