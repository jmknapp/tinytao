"""Level-4 measurements on exact H=1 Model C populations, any prime."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from src.analysis.fourier import discrete_log_table, primitive_root
from src.analysis.group_law import (
    TWO_PI,
    apply_scale_rot,
    centered_hom_deg_batch,
    exact_counts,
    exact_unit_circle,
    fit_dlog_circle_batch,
    homomorphism_delta,
    lstsq_readout,
    pair_masks,
    radius_stats,
    restore_params,
    snapshot_params,
    split_accuracies,
    summarize_delta,
    unit_normalize_learned,
    units_mod,
    winding_readout,
    wrap,
    write_exact_embeddings,
)
from src.population.complex import PopulationComplexMLP
from src.symbolic_f13 import classify_pairs, from_fit


def effective_winding(k: np.ndarray, conjugate: np.ndarray, n: int) -> np.ndarray:
    """Fold the conjugate flag into k so k and -k are distinct windings."""
    k = np.asarray(k, dtype=np.int64)
    conj = np.asarray(conjugate, dtype=bool)
    return np.where(conj, np.mod(-k, n), np.mod(k, n)).astype(np.int64)


def winding_table(k_eff: np.ndarray, n: int) -> list[dict[str, Any]]:
    units = set(units_mod(n))
    rows = []
    for k in range(n):
        count = int((k_eff == k).sum())
        rows.append(
            {
                "k": k,
                "count": count,
                "gcd": math.gcd(k, n) if k else n,
                "unit": k in units,
                "exact_generalizer_count": count,
            }
        )
    return rows


def _angular_error_deg(xy: np.ndarray, k_eff: np.ndarray, phi: np.ndarray, dlog: np.ndarray, n: int) -> np.ndarray:
    dlog_vals = np.array([int(dlog[i + 1]) for i in range(n)], dtype=np.float64)
    theta_obs = np.arctan2(xy[:, :, 1], xy[:, :, 0])
    theta_pred = phi[:, None] + TWO_PI * k_eff[:, None] * dlog_vals[None, :] / n
    return np.degrees(np.abs(wrap(theta_obs - theta_pred)).mean(axis=1))


def analyze_h1_exact(
    model: PopulationComplexMLP,
    exact_mask: torch.Tensor,
    domain_x: torch.Tensor,
    domain_y: torch.Tensor,
    train_idx: torch.Tensor,
    test_idx: torch.Tensor,
    p: int,
    held_residue: int,
) -> dict[str, Any]:
    n = p - 1
    units = set(units_mod(n))
    dlog = discrete_log_table(p)
    n_exact = int(exact_mask.sum())
    if n_exact == 0:
        return {
            "n_exact": 0,
            "n_analyzed": 0,
            "primitive_root": int(primitive_root(p)),
            "winding_table": winding_table(np.zeros(0, dtype=np.int64), n),
            "n_observed_faithful_families": 0,
            "phi": len(units),
        }

    idx = torch.where(exact_mask.detach().cpu())[0]
    E = model.E.detach().cpu()[idx, :, 0, :].numpy().astype(np.float64)
    batch = fit_dlog_circle_batch(E, dlog, n)
    k_eff = effective_winding(batch["k"], batch["conjugate"], n)
    train_mask = pair_masks(p, held_residue, "a")["train"]
    hom_rows = []
    rad_rows = []
    for i in range(n_exact):
        delta = homomorphism_delta(E[i], p)
        hom_rows.append(summarize_delta(delta, train_mask))
        rad_rows.append(radius_stats(E[i], dlog, n))
    unit_flag = np.array([int(ki) in units for ki in k_eff])
    ang = _angular_error_deg(E, k_eff, batch["phi"], dlog, n)
    k_counts = {int(v): int((k_eff == v).sum()) for v in sorted(set(k_eff.tolist()))}
    observed_units = sorted({int(v) for v in k_eff.tolist() if int(v) in units})
    observed_non_units = sorted({int(v) for v in k_eff.tolist() if int(v) not in units})

    sub = PopulationComplexMLP(
        n_exact,
        2 * n,
        [1],
        n,
        device=model.E.device,
        dtype=model.E.dtype,
    )
    with torch.no_grad():
        sub.E.copy_(model.E.detach()[idx])
        sub.W_out.copy_(model.W_out.detach()[idx])
        sub.b_out.copy_(model.b_out.detach()[idx])
    snap = snapshot_params(sub)
    fits = []
    for i in range(n_exact):
        unit = exact_unit_circle(int(batch["k"][i]), dlog, n, bool(batch["conjugate"][i]))
        fitted = apply_scale_rot(unit, float(batch["scale"][i]), float(batch["phi"][i]))
        fits.append(
            [
                {
                    "k": int(batch["k"][i]),
                    "phi": float(batch["phi"][i]),
                    "scale": float(batch["scale"][i]),
                    "conjugate": bool(batch["conjugate"][i]),
                    "fitted": fitted,
                    "r2": float(batch["r2"][i]),
                }
            ]
        )

    def _eval():
        acc = split_accuracies(sub, domain_x, domain_y, train_idx, test_idx)
        return {name: exact_counts(t) for name, t in acc.items()}

    variants = {}
    restore_params(sub, snap)
    variants["identity"] = _eval()
    restore_params(sub, snap)
    write_exact_embeddings(sub, fits, "aligned")
    variants["A_aligned_exact_frozen_W"] = _eval()
    restore_params(sub, snap)
    write_exact_embeddings(sub, fits, "strict_unit")
    lstsq_readout(sub, domain_x, domain_y)
    variants["B_unit_roots_LS_W"] = _eval()
    restore_params(sub, snap)
    write_exact_embeddings(sub, fits, "aligned")
    with torch.no_grad():
        for i, net_fits in enumerate(fits):
            W, b = winding_readout(net_fits[0], dlog, n)
            sub.W_out[i] = torch.as_tensor(W, device=sub.W_out.device, dtype=sub.W_out.dtype)
            sub.b_out[i] = torch.as_tensor(b, device=sub.b_out.device, dtype=sub.b_out.dtype)
    variants["C_aligned_geometric_W"] = _eval()
    restore_params(sub, snap)
    with torch.no_grad():
        rmean = sub.E.norm(dim=-1).mean(dim=(1, 2))
        unit_normalize_learned(sub)
        sub.W_out.mul_(rmean.reshape(n_exact, 1, 1) ** 2)
    variants["D_unit_norm_scale_matched_W"] = _eval()
    restore_params(sub, snap)

    a = torch.arange(1, p, device=domain_x.device).repeat_interleave(n).cpu().numpy()
    b = torch.arange(1, p, device=domain_x.device).repeat(n).cpu().numpy()
    y = domain_y.cpu().numpy()
    sym_ok = np.zeros(n_exact, dtype=bool)
    for i, net_fits in enumerate(fits):
        cert = from_fit(p, net_fits[0])
        pred = classify_pairs(cert, a, b)
        sym_ok[i] = bool((pred == y).all())
    n_sym = int(sym_ok.sum())

    def _mean(key, mask=None):
        vals = [row[key] for i, row in enumerate(hom_rows) if mask is None or mask[i]]
        return float(np.mean(vals)) if vals else float("nan")

    def _subset(mask: np.ndarray) -> dict[str, Any]:
        m = np.asarray(mask, dtype=bool)
        n_m = int(m.sum())
        if n_m == 0:
            return {"n": 0}
        return {
            "n": n_m,
            "mean_r2": float(batch["r2"][m].mean()),
            "min_r2": float(batch["r2"][m].min()),
            "mean_angular_err_deg": float(ang[m].mean()),
            "mean_radius_cv": float(np.mean([rad_rows[i]["cv"] for i in range(n_exact) if m[i]])),
            "mean_abs_centered_deg": _mean("mean_abs_centered_deg", m),
            "rms_centered_deg": float(
                np.degrees(np.mean([hom_rows[i]["rms_centered_rad"] for i in range(n_exact) if m[i]]))
            ),
            "mean_abs_centered_train_deg": float(
                np.degrees(np.mean([hom_rows[i]["mean_abs_centered_train_rad"] for i in range(n_exact) if m[i]]))
            ),
            "mean_abs_centered_held_deg": float(
                np.degrees(np.mean([hom_rows[i]["mean_abs_centered_held_rad"] for i in range(n_exact) if m[i]]))
            ),
            "frac_max_lt_2deg": float(
                np.mean([hom_rows[i]["max_abs_centered_deg"] < 2.0 for i in range(n_exact) if m[i]])
            ),
            "n_symbolic_exact": int(sym_ok[m].sum()),
            "frac_symbolic_exact": float(sym_ok[m].mean()),
            "k_counts": {int(v): int((k_eff[m] == v).sum()) for v in sorted(set(k_eff[m].tolist()))},
        }

    frac_A = variants["A_aligned_exact_frozen_W"]["domain"]["frac_exact"]
    mean_hom = _mean("mean_abs_centered_deg")
    mean_hom_train = float(np.degrees(np.mean([row["mean_abs_centered_train_rad"] for row in hom_rows])))
    mean_hom_held = float(np.degrees(np.mean([row["mean_abs_centered_held_rad"] for row in hom_rows])))
    unit_sub = _subset(unit_flag)
    nonunit_sub = _subset(~unit_flag)
    mechanism = bool(
        unit_sub.get("n", 0) >= 1
        and unit_sub.get("frac_symbolic_exact", 0) >= 0.99
        and unit_sub.get("mean_abs_centered_deg", 99) < 2.0
    )
    exclusive_units = bool(n_exact > 0 and float(unit_flag.mean()) >= 0.99)
    return {
        "n_exact": n_exact,
        "n_analyzed": n_exact,
        "primitive_root": int(primitive_root(p)),
        "fit": {
            "mean_r2": float(batch["r2"].mean()),
            "min_r2": float(batch["r2"].min()),
            "median_r2": float(np.median(batch["r2"])),
            "k_counts": k_counts,
            "k_raw_counts": {int(v): int((batch["k"] == v).sum()) for v in sorted(set(batch["k"].tolist()))},
            "n_unit_k": int(unit_flag.sum()),
            "frac_unit_k": float(unit_flag.mean()),
            "observed_units": observed_units,
            "observed_non_units": observed_non_units,
            "frac_conjugate": float(batch["conjugate"].mean()),
            "mean_angular_err_deg": float(ang.mean()),
            "max_angular_err_deg": float(ang.max()),
            "mean_radius_cv": float(np.mean([r["cv"] for r in rad_rows])),
        },
        "winding_table": winding_table(k_eff, n),
        "n_observed_faithful_families": len(observed_units),
        "phi": len(units),
        "hom": {
            "mean_abs_centered_deg": mean_hom,
            "rms_centered_deg": float(np.degrees(np.mean([row["rms_centered_rad"] for row in hom_rows]))),
            "mean_max_centered_deg": _mean("max_abs_centered_deg"),
            "mean_abs_centered_train_deg": mean_hom_train,
            "mean_abs_centered_held_deg": mean_hom_held,
            "frac_max_lt_2deg": float(np.mean([row["max_abs_centered_deg"] < 2.0 for row in hom_rows])),
            "mean_hom_deg_batch": float(centered_hom_deg_batch(E, p).mean()),
        },
        "substitutions": variants,
        "n_symbolic_program_exact": n_sym,
        "frac_symbolic_program_exact": n_sym / n_exact,
        "frac_A_still_exact": frac_A,
        "frac_B_still_exact": variants["B_unit_roots_LS_W"]["domain"]["frac_exact"],
        "frac_C_still_exact": variants["C_aligned_geometric_W"]["domain"]["frac_exact"],
        "frac_D_still_exact": variants["D_unit_norm_scale_matched_W"]["domain"]["frac_exact"],
        "unit_exact": unit_sub,
        "nonunit_exact": nonunit_sub,
        "level4_mechanism_present": mechanism,
        "level4_exclusive_units": exclusive_units,
        "level4_replication": bool(mechanism),
    }


def analyze_failures_h1(model: PopulationComplexMLP, exact_mask: torch.Tensor, p: int) -> dict[str, Any]:
    fail = ~exact_mask
    n_fail = int(fail.sum())
    n = p - 1
    units = set(units_mod(n))
    if n_fail == 0:
        return {"n_fail": 0, "winding_table": winding_table(np.zeros(0, dtype=np.int64), n)}
    dlog = discrete_log_table(p)
    idx = torch.where(fail.detach().cpu())[0]
    E = model.E.detach().cpu()[idx, :, 0, :].numpy().astype(np.float64)
    batch = fit_dlog_circle_batch(E, dlog, n)
    k_eff = effective_winding(batch["k"], batch["conjugate"], n)
    unit_flag = np.array([int(ki) in units for ki in k_eff])
    k_counts = {int(v): int((k_eff == v).sum()) for v in sorted(set(k_eff.tolist()))}
    r2 = batch["r2"]
    hom = centered_hom_deg_batch(E, p)
    cv = np.array([radius_stats(E[i], dlog, n)["cv"] for i in range(n_fail)])
    cats = {
        "non_unit_winding": int(((~unit_flag) & (r2 >= 0.85)).sum()),
        "unit_winding_but_inexact": int((unit_flag & (r2 >= 0.85)).sum()),
        "distorted_circle": int((r2 < 0.85).sum()),
        "high_radius_cv": int((cv > 0.3).sum()),
        "high_hom_error": int((hom > 5.0).sum()),
    }
    return {
        "n_fail": n_fail,
        "mean_r2": float(r2.mean()),
        "frac_unit_k": float(unit_flag.mean()),
        "k_counts": k_counts,
        "mean_hom_deg": float(hom.mean()),
        "mean_radius_cv": float(cv.mean()),
        "categories": cats,
        "winding_table": winding_table(k_eff, n),
        "observed_units": sorted({int(v) for v in k_eff.tolist() if int(v) in units}),
        "observed_non_units": sorted({int(v) for v in k_eff.tolist() if int(v) not in units}),
    }
