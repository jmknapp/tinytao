"""Non-unit exact H=1 solvers: fibers, radius, product geometry, readout.

Measurements only. Hypotheses live in stage_a5_hypotheses.py and must be
written to a run before this module is used for interpretation.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from src.analysis.fourier import discrete_log_table, primitive_root
from src.analysis.group_law import (
    TWO_PI,
    circ_mean,
    fit_dlog_circle_batch,
    homomorphism_delta,
    pair_masks,
    radius_stats,
    summarize_delta,
    units_mod,
    wrap,
)
from src.analysis.level4_h1 import effective_winding
from src.symbolic_f13 import classify_pairs, from_fit


def gcd_k(k: int, n: int) -> int:
    return math.gcd(int(k), int(n)) if int(k) else int(n)


def kernel_size(k: int, n: int) -> int:
    return gcd_k(k, n)


def n_phases(k: int, n: int) -> int:
    d = kernel_size(k, n)
    return n // d


def ideal_phase(k: int, dlog_vals: np.ndarray, n: int) -> np.ndarray:
    return TWO_PI * (int(k) % n) * dlog_vals / n


def fiber_id_from_k(k: int, dlog_vals: np.ndarray, n: int) -> np.ndarray:
    """Elements with the same χ_k phase: dlog equal mod (n/gcd(k,n))."""
    q = n_phases(k, n)
    return np.mod(dlog_vals.astype(np.int64), q)


def fiber_members(k: int, p: int) -> dict[int, list[int]]:
    n = p - 1
    dlog = discrete_log_table(p)
    dlog_vals = np.array([int(dlog[a]) for a in range(1, p)], dtype=np.int64)
    fid = fiber_id_from_k(k, dlog_vals, n)
    out: dict[int, list[int]] = {}
    for a in range(1, p):
        f = int(fid[a - 1])
        out.setdefault(f, []).append(a)
    return dict(sorted(out.items()))


def winding_scores(xy: np.ndarray, dlog: np.ndarray, n: int) -> dict[str, np.ndarray]:
    """Best R² per k (max over conjugation). xy [P, n, 2]."""
    pop = xy.shape[0]
    y = np.concatenate([xy[:, :, 0], xy[:, :, 1]], axis=1)
    y_var = ((xy - xy.mean(axis=1, keepdims=True)) ** 2).sum(axis=(1, 2))
    y_var = np.maximum(y_var, 1e-18)
    theta0 = TWO_PI * np.array([int(dlog[i + 1]) for i in range(n)], dtype=np.float64) / n
    r2_k = np.full((pop, n), -1.0)
    conj_k = np.zeros((pop, n), dtype=bool)
    for k in range(n):
        best = np.full(pop, -1.0)
        best_conj = np.zeros(pop, dtype=bool)
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
            better = r2 > best
            best = np.where(better, r2, best)
            best_conj = np.where(better, conj, best_conj)
        r2_k[:, k] = best
        conj_k[:, k] = best_conj
    return {"r2_k": r2_k, "conj_k": conj_k}


def classify_winding(
    k_eff: int,
    n: int,
    r2: float,
    r2_second: float,
    second_unit: bool,
    r2_clear: float,
    r2_gap: float,
) -> str:
    """Winding class. Constant-radius R² is a diagnostic, not a gate for non-units.

    A radial quotient code is expected to fit a constant-radius character only
    moderately well. Gating non-unit membership on R²>=0.85 would hide the
    hypothesis under test.
    """
    units = set(units_mod(n))
    is_unit = int(k_eff) in units
    if (r2 - r2_second) < r2_gap and (is_unit != bool(second_unit)):
        return "ambiguous"
    if is_unit:
        return "faithful_unit" if r2 >= r2_clear else "ambiguous"
    return "non_unit"


def entropy_from_labels(y: np.ndarray, groups: np.ndarray) -> float:
    """H(Y | group) in bits. y and groups length N."""
    h = 0.0
    n = len(y)
    for g in np.unique(groups):
        mask = groups == g
        m = int(mask.sum())
        if m == 0:
            continue
        _, counts = np.unique(y[mask], return_counts=True)
        p = counts / counts.sum()
        ent = float(-(p * np.log2(np.maximum(p, 1e-15))).sum())
        h += (m / n) * ent
    return h


def product_hidden(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """All pairs. xy [n,2] -> h [n,n,2], r [n,n], theta [n,n]."""
    za = xy[:, None, :]
    zb = xy[None, :, :]
    h = np.stack(
        [
            za[..., 0] * zb[..., 0] - za[..., 1] * zb[..., 1],
            za[..., 0] * zb[..., 1] + za[..., 1] * zb[..., 0],
        ],
        axis=-1,
    )
    r = np.linalg.norm(h, axis=-1)
    theta = np.arctan2(h[..., 1], h[..., 0])
    return h, r, theta


def analyze_one_net(
    xy: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    p: int,
    k_eff: int,
    phi: float,
    r2: float,
    conjugate: bool,
    scale: float,
    r2_k: np.ndarray,
    held_residue: int,
    census: dict[str, float],
) -> dict[str, Any]:
    n = p - 1
    dlog = discrete_log_table(p)
    dlog_vals = np.array([int(dlog[a]) for a in range(1, p)], dtype=np.int64)
    residues = np.arange(1, p)
    units = set(units_mod(n))
    d = kernel_size(k_eff, n)
    q = n_phases(k_eff, n)
    fid = fiber_id_from_k(k_eff, dlog_vals, n)
    r = np.linalg.norm(xy, axis=-1)
    theta = np.arctan2(xy[:, 1], xy[:, 0])
    theta_ideal = phi + ideal_phase(k_eff, dlog_vals, n)
    ang_res = float(np.degrees(np.abs(wrap(theta - theta_ideal)).mean()))
    train_mask = pair_masks(p, held_residue, "a")["train"]
    hom = summarize_delta(homomorphism_delta(xy, p), train_mask)
    rad = radius_stats(xy, dlog, n)

    order = np.argsort(r2_k)[::-1]
    k_best = int(order[0])
    k_second = int(order[1])
    r2_second = float(r2_k[k_second])
    kind = classify_winding(
        int(k_eff),
        n,
        float(r2),
        r2_second,
        k_second in units,
        census["r2_clear"],
        census["r2_gap"],
    )

    fibers = fiber_members(int(k_eff), p)
    fiber_rows = []
    within_sep = []
    rank_vs_dlog_mod = {m: [] for m in range(2, n + 1)}
    rank_vs_res_mod = {m: [] for m in range(2, p)}
    rank_vs_dlog_parity = []
    for f, members in fibers.items():
        idx = [a - 1 for a in members]
        rr = r[idx]
        order_r = np.argsort(rr)
        ranked = [members[int(i)] for i in order_r]  # small to large radius
        ratio = float(rr.max() / (rr.min() + 1e-12))
        sep = float((rr.max() - rr.min()) / (rr.mean() + 1e-12))
        within_sep.append(sep)
        dlogs = dlog_vals[idx]
        fiber_rows.append(
            {
                "fiber": int(f),
                "residues": members,
                "dlogs": [int(v) for v in dlogs],
                "radii": [float(v) for v in rr],
                "small_radius_residue": int(ranked[0]),
                "large_radius_residue": int(ranked[-1]),
                "radius_ratio": ratio,
                "radius_sep": sep,
            }
        )
        if len(members) == 2:
            # 1 if larger-radius residue has odd dlog, else 0
            large = ranked[-1]
            rank_vs_dlog_parity.append(int(dlog[large] % 2))
            for m in rank_vs_dlog_mod:
                rank_vs_dlog_mod[m].append(int(dlog[large] % m))
            for m in rank_vs_res_mod:
                rank_vs_res_mod[m].append(int(large % m))

    # Simple descriptors: does "large radius" equal a function, up to a global bit?
    descriptor_hits = {}
    if len(next(iter(fibers.values()))) == 2:
        large = np.array([row["large_radius_residue"] for row in fiber_rows])
        small = np.array([row["small_radius_residue"] for row in fiber_rows])
        for name, fn in (
            ("dlog_parity_of_large", lambda a: int(dlog[int(a)]) % 2),
            ("dlog_mod_d_of_large", lambda a: int(dlog[int(a)]) % max(d, 1)),
            ("residue_mod_2", lambda a: int(a) % 2),
            ("is_negation_large_equals_minus_small", lambda a: None),
        ):
            if name.startswith("is_negation"):
                # fibers should be {a,-a} when d=2 and n even
                ok = all((p - s) % p == L or (p - L) % p == s for s, L in zip(small, large))
                descriptor_hits["fibers_are_a_and_minus_a"] = bool(ok)
                continue
            vals = np.array([fn(a) for a in large])
            # constant after global flip: majority
            if vals.size:
                maj = int(np.round(vals.mean())) if set(vals.tolist()) <= {0, 1} else None
                if maj is not None:
                    descriptor_hits[name] = {
                        "frac_match_majority": float(np.mean(vals == maj)),
                        "majority_value": maj,
                    }
                else:
                    u, c = np.unique(vals, return_counts=True)
                    descriptor_hits[name] = {
                        "mode": int(u[int(c.argmax())]),
                        "frac_mode": float(c.max() / c.sum()),
                    }

    # Product geometry
    h, pr, pth = product_hidden(xy)
    aa = residues[:, None] * np.ones((1, n), dtype=np.int64)
    bb = residues[None, :] * np.ones((n, 1), dtype=np.int64)
    target = (aa * bb) % p  # residues 1..p-1
    target_idx = target - 1
    # Ideal product phase class of ab: fiber of ab
    tgt_fiber = fid[target_idx]
    learned_prod_fiber = _assign_nearest_fiber(pth, k_eff, phi, dlog_vals, n, q)
    quotient_acc = float((learned_prod_fiber == tgt_fiber).mean())

    # Phase-only target: canonical residue in fiber (even dlog if unique, else min residue)
    canon = _canonical_in_fiber(fibers, dlog)
    pred_phase = np.vectorize(lambda f: canon[int(f)])(learned_prod_fiber)
    phase_only_acc = float((pred_phase == target).mean())

    # Radius-only: 1-NN to class-mean product radius (in-sample)
    class_mean_r = np.array([pr[target == c].mean() if (target == c).any() else 0.0 for c in residues])
    pred_r = residues[np.abs(pr[..., None] - class_mean_r[None, None, :]).argmin(axis=-1)]
    radius_only_acc = float((pred_r == target).mean())

    # Joint: within each learned phase bin, 1-NN on product radius to class means in that bin
    pred_joint = np.empty((n, n), dtype=np.int64)
    radius_resolves = []
    for f in range(q):
        mask = learned_prod_fiber == f
        if not mask.any():
            continue
        classes_here = np.unique(target[mask])
        means = {int(c): float(pr[mask & (target == c)].mean()) for c in classes_here if (mask & (target == c)).any()}
        rr = pr[mask]
        # uniqueness: interval overlap of product radii per true class in this bin
        intervals = {}
        for c in classes_here:
            vals = pr[mask & (target == c)]
            if vals.size:
                intervals[int(c)] = (float(vals.min()), float(vals.max()))
        overlap = _interval_overlap(intervals)
        radius_resolves.append({"fiber": int(f), "n_classes": int(len(classes_here)), "overlapping_classes": overlap})
        # decode
        keys = list(means.keys())
        mvals = np.array([means[k] for k in keys])
        pick = np.abs(rr[:, None] - mvals[None, :]).argmin(axis=1)
        pred_joint[mask] = np.array(keys)[pick]
    joint_acc = float((pred_joint == target).mean())

    # Discrete radial state: rank within fiber of embeddings (d levels)
    # For d=2, bit = r > median of the two
    embed_bit = np.zeros(n, dtype=np.int64)
    for f, members in fibers.items():
        idx = np.array([a - 1 for a in members])
        order_r = np.argsort(r[idx])
        for rank, j in enumerate(order_r):
            embed_bit[idx[int(j)]] = rank

    # Product bit from embedding bits: for d=2, mixed vs same
    if d == 2:
        bit_a = embed_bit[:, None]
        bit_b = embed_bit[None, :]
        mixed = (bit_a != bit_b).astype(np.int64)
        # target dlog parity
        tgt_parity = (dlog_vals[target_idx]) % 2
        # Does mixed predict target parity? (up to flip)
        acc_m = float((mixed == tgt_parity).mean())
        acc_m_flip = float((mixed != tgt_parity).mean())
        mixed_parity_acc = max(acc_m, acc_m_flip)
        mixed_parity_polarity = 1 if acc_m >= acc_m_flip else -1
    else:
        mixed_parity_acc = float("nan")
        mixed_parity_polarity = 0

    groups_phase = learned_prod_fiber.reshape(-1)
    y_flat = target.reshape(-1)
    # radius state: above vs below median product radius inside each phase bin
    rad_state = np.zeros((n, n), dtype=np.int64)
    for f in range(q):
        mask = learned_prod_fiber == f
        if not mask.any():
            continue
        med = np.median(pr[mask])
        rad_state[mask] = (pr[mask] > med).astype(np.int64)
    groups_joint = groups_phase * 10 + rad_state.reshape(-1)

    # Readout polar geometry of W columns [2, C]
    W2 = np.asarray(W, dtype=np.float64)
    if W2.shape[0] != 2:
        W2 = W2.T if W2.shape[-1] == 2 else W2
    w_xy = W2.T if W2.shape[0] == 2 else W2  # [C, 2]
    if w_xy.shape[1] != 2:
        w_xy = W2.reshape(2, n).T
    w_ang = np.arctan2(w_xy[:, 1], w_xy[:, 0])
    w_rad = np.linalg.norm(w_xy, axis=1)
    # Frozen readout on unit-normalized h and on radius-equalized h (diagnostics, not embedding ablations)
    logits = h @ w_xy.T + np.asarray(b).reshape(1, 1, -1)
    pred_net = logits.argmax(axis=-1) + 1
    net_acc = float((pred_net == target).mean())
    hn = h / np.clip(np.linalg.norm(h, axis=-1, keepdims=True), 1e-12, None)
    pred_unit_h = (hn @ w_xy.T + np.asarray(b).reshape(1, 1, -1)).argmax(axis=-1) + 1
    unit_h_acc = float((pred_unit_h == target).mean())
    rmean = float(np.linalg.norm(h, axis=-1).mean())
    hs = hn * rmean
    pred_shell = (hs @ w_xy.T + np.asarray(b).reshape(1, 1, -1)).argmax(axis=-1) + 1
    mean_r_h_acc = float((pred_shell == target).mean())

    # W columns: angle gap within fibers vs across
    fiber_w_ang_spread = []
    fiber_w_rad_ratio = []
    for f, members in fibers.items():
        idx = [a - 1 for a in members]
        angs = w_ang[idx]
        circ = wrap(angs - circ_mean(angs))
        fiber_w_ang_spread.append(float(np.degrees(np.abs(circ).mean())))
        wr = w_rad[idx]
        fiber_w_rad_ratio.append(float(wr.max() / (wr.min() + 1e-12)))

    # Symbolic faithful-character replacement (should fail for non-units)
    a_flat = np.repeat(residues, n)
    b_flat = np.tile(residues, n)
    y_class = ((a_flat * b_flat) % p) - 1
    cert = from_fit(p, {"k": int(k_eff), "phi": float(phi), "scale": float(scale), "conjugate": False})
    pred_sym = classify_pairs(cert, a_flat, b_flat)
    symbolic_exact = bool((pred_sym == y_class).all())

    return {
        "k_eff": int(k_eff),
        "k_raw_best_by_r2": k_best,
        "k_second": k_second,
        "r2": float(r2),
        "r2_second": r2_second,
        "gcd": d,
        "n_phases": q,
        "kernel_size": d,
        "kind": kind,
        "unit": int(k_eff) in units,
        "phi": float(phi),
        "scale": float(scale),
        "conjugate": bool(conjugate),
        "angular_residual_deg": ang_res,
        "radius_cv": float(rad["cv"]),
        "mean_abs_centered_deg": hom["mean_abs_centered_deg"],
        "mean_abs_centered_train_deg": float(np.degrees(hom["mean_abs_centered_train_rad"])),
        "mean_abs_centered_held_deg": float(np.degrees(hom["mean_abs_centered_held_rad"])),
        "max_abs_centered_deg": hom["max_abs_centered_deg"],
        "symbolic_roots_exact": symbolic_exact,
        "fibers": fiber_rows,
        "mean_within_fiber_sep": float(np.mean(within_sep)) if within_sep else float("nan"),
        "min_within_fiber_sep": float(np.min(within_sep)) if within_sep else float("nan"),
        "frac_fibers_radius_ratio_gt_1_2": float(np.mean([row["radius_ratio"] > 1.2 for row in fiber_rows])),
        "descriptor_hits": descriptor_hits,
        "quotient_acc_from_product_phase": quotient_acc,
        "target_acc_phase_only": phase_only_acc,
        "target_acc_radius_only": radius_only_acc,
        "target_acc_phase_and_radius": joint_acc,
        "H_target": float(np.log2(n)),
        "H_target_given_phase": entropy_from_labels(y_flat, groups_phase),
        "H_target_given_phase_radiusbin": entropy_from_labels(y_flat, groups_joint),
        "mixed_vs_target_dlog_parity_acc": mixed_parity_acc,
        "radius_interval_overlap_by_fiber": radius_resolves,
        "frac_phase_bins_radius_disjoint": float(
            np.mean([len(row["overlapping_classes"]) == 0 for row in radius_resolves])
        )
        if radius_resolves
        else float("nan"),
        "net_acc_check": net_acc,
        "unit_normalize_h_frozen_W_acc": unit_h_acc,
        "mean_radius_h_frozen_W_acc": mean_r_h_acc,
        "mean_W_angle_spread_within_fiber_deg": float(np.mean(fiber_w_ang_spread)) if fiber_w_ang_spread else float("nan"),
        "mean_W_radius_ratio_within_fiber": float(np.mean(fiber_w_rad_ratio)) if fiber_w_rad_ratio else float("nan"),
        "embed_radius": [float(v) for v in r],
        "embed_theta": [float(v) for v in theta],
        "xy": xy.tolist(),
        "W_xy": w_xy.tolist(),
        "b": np.asarray(b, dtype=np.float64).reshape(-1).tolist(),
    }


def _canonical_in_fiber(fibers: dict[int, list[int]], dlog: np.ndarray) -> dict[int, int]:
    out = {}
    for f, members in fibers.items():
        even = [a for a in members if int(dlog[a]) % 2 == 0]
        out[int(f)] = int(even[0]) if even else int(min(members))
    return out


def _assign_nearest_fiber(theta: np.ndarray, k: int, phi: float, dlog_vals: np.ndarray, n: int, q: int) -> np.ndarray:
    """Nearest of the q ideal product phases (global phase 2φ because h ~ z z)."""
    # Ideal phase of residue a is phi + 2π k dlog / n. Product of two embeddings
    # has phase 2φ + 2π k (dlog a + dlog b)/n = 2φ + ideal_phase(ab)  (mod the k-winding).
    # Representative residues: one per fiber (dlog = 0..q-1).
    reps = np.arange(q, dtype=np.float64)  # dlog mod q
    # A residue with dlog ≡ s (mod q) has ideal theta = phi + 2π k s / n
    # Product: 2 phi + 2π k (s_a+s_b)/n. Fiber of product is (s_a+s_b) mod q,
    # whose representative dlog = that value (0..q-1).
    ideal = 2.0 * phi + TWO_PI * int(k) * reps / n
    delta = wrap(theta[..., None] - ideal.reshape(1, 1, q))
    return np.abs(delta).argmin(axis=-1)


def _interval_overlap(intervals: dict[int, tuple[float, float]]) -> list[list[int]]:
    keys = list(intervals.keys())
    bad = []
    for i, c1 in enumerate(keys):
        a1, b1 = intervals[c1]
        for c2 in keys[i + 1 :]:
            a2, b2 = intervals[c2]
            if not (b1 < a2 or b2 < a1):
                bad.append([int(c1), int(c2)])
    return bad


def population_descriptor_consistency(rows: list[dict[str, Any]], p: int, k: int) -> dict[str, Any]:
    """After a global high/low flip per net, does large-radius match dlog parity?"""
    dlog = discrete_log_table(p)
    fibers = fiber_members(k, p)
    if not rows or any(len(v) != 2 for v in fibers.values()):
        return {"n": len(rows), "applicable": False}
    n_match = 0
    n_all_fibers = 0
    polarities = []
    for row in rows:
        votes = []
        for fr in row["fibers"]:
            large = fr["large_radius_residue"]
            votes.append(int(dlog[large]) % 2)
        if not votes:
            continue
        maj = 1 if sum(votes) >= (len(votes) / 2) else 0
        polarities.append(maj)
        n_match += int(sum(v == maj for v in votes))
        n_all_fibers += len(votes)
    return {
        "n": len(rows),
        "applicable": True,
        "frac_fibers_large_radius_has_consistent_dlog_parity": n_match / max(n_all_fibers, 1),
        "frac_nets_all_fibers_agree": float(
            np.mean(
                [
                    all(int(dlog[fr["large_radius_residue"]]) % 2 == (1 if sum(int(dlog[x["large_radius_residue"]]) % 2 for x in row["fibers"]) >= len(row["fibers"]) / 2 else 0) for fr in row["fibers"])
                    for row in rows
                ]
            )
        )
        if rows
        else 0.0,
        "mean_polarity_even_is_large": float(np.mean([1 - v for v in polarities])) if polarities else float("nan"),
    }
