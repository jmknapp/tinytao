"""Mechanistic probes for Model C 2-D embeddings.

Measurements only. A high circle R² is not an algorithm claim;
substitution and homomorphism error are the causal tests.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from src.analysis.fourier import discrete_log_table
from src.population.complex import PopulationComplexMLP
from src.tasks.modular import ModularMultiplicationStar


def _angles_and_radii(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    radii = np.linalg.norm(xy, axis=-1)
    angles = np.arctan2(xy[:, 1], xy[:, 0])
    return angles, radii


def fit_dlog_circle(xy: np.ndarray, dlog: np.ndarray, n: int) -> dict[str, Any]:
    """Best scaled, rotated (and optionally conjugated) dlog sinusoid in R^2.

    xy: [n, 2] embedding of residues 1..n (index i = residue i+1).
    """
    theta = 2.0 * math.pi * np.array([int(dlog[i + 1]) for i in range(n)], dtype=np.float64) / n
    best = {"r2": -1.0}
    y = np.concatenate([xy[:, 0], xy[:, 1]])
    y_var = float(((xy - xy.mean(axis=0)) ** 2).sum())
    if y_var < 1e-18:
        return {"r2": 0.0, "k": 0, "phi": 0.0, "scale": 0.0, "conjugate": False, "fitted": np.zeros_like(xy)}
    for k in range(0, n):
        for conj in (False, True):
            c = np.cos(k * theta)
            s = np.sin(k * theta)
            if conj:
                s = -s
            # [x,y] ≈ scale * R(phi) @ [c, s] = [a c - b s, b c + a s]
            # a = scale cos phi, b = scale sin phi
            Bx = np.column_stack([c, -s])
            By = np.column_stack([s, c])
            B = np.vstack([Bx, By])
            coef, *_ = np.linalg.lstsq(B, y, rcond=None)
            a, b = float(coef[0]), float(coef[1])
            fitted = np.stack([a * c - b * s, b * c + a * s], axis=1)
            ss_res = float(((xy - fitted) ** 2).sum())
            r2 = 1.0 - ss_res / y_var
            if r2 > best["r2"]:
                scale = math.hypot(a, b)
                phi = math.atan2(b, a)
                best = {
                    "r2": r2,
                    "k": k,
                    "phi": phi,
                    "scale": scale,
                    "conjugate": conj,
                    "fitted": fitted,
                    "a": a,
                    "b": b,
                }
    return best


def homomorphism_errors(xy: np.ndarray, n: int) -> dict[str, float]:
    """Compare complex product e(a)e(b) to e(ab) on F_p* (p=n+1)."""
    p = n + 1
    e = xy[:, 0] + 1j * xy[:, 1]
    prod_err = []
    ang_err = []
    for ia in range(n):
        a = ia + 1
        for ib in range(n):
            b = ib + 1
            ic = ((a * b) % p) - 1
            z = e[ia] * e[ib]
            w = e[ic]
            prod_err.append(abs(z - w))
            if abs(z) > 1e-8 and abs(w) > 1e-8:
                ang_err.append(abs(np.angle(z / w)))
    ang = np.array(ang_err) if ang_err else np.array([math.pi])
    ang = np.minimum(ang, 2 * math.pi - ang)
    return {
        "mean_product_abs_err": float(np.mean(prod_err)),
        "median_product_abs_err": float(np.median(prod_err)),
        "mean_angle_err_rad": float(np.mean(ang)),
        "median_angle_err_rad": float(np.median(ang)),
    }


@torch.no_grad()
def substitute_fitted_circle(
    model: PopulationComplexMLP,
    fits: list[dict[str, Any]],
    domain_x: torch.Tensor,
    domain_y: torch.Tensor,
) -> dict[str, float]:
    """Replace embeddings by fitted dlog circles; freeze readout."""
    orig = model.E.detach().clone()
    for i, fit in enumerate(fits):
        xy = fit["fitted"]
        model.E[i, :, 0, 0] = torch.as_tensor(xy[:, 0], device=model.E.device, dtype=model.E.dtype)
        model.E[i, :, 0, 1] = torch.as_tensor(xy[:, 1], device=model.E.device, dtype=model.E.dtype)
    logits = model(domain_x)
    acc = (logits.argmax(dim=-1) == domain_y.unsqueeze(1)).float().mean(dim=0)
    model.E.copy_(orig)
    exact = acc >= 1.0 - 1e-12
    return {
        "n": int(acc.numel()),
        "n_exact": int(exact.sum()),
        "mean_domain_acc": float(acc.mean()),
        "max_domain_acc": float(acc.max()),
        "min_domain_acc": float(acc.min()),
    }


@torch.no_grad()
def analyze_complex_checkpoint(
    ckpt: dict[str, torch.Tensor],
    device: torch.device,
    p: int = 13,
) -> dict[str, Any]:
    n = p - 1
    E = ckpt["E"]
    pop = int(E.shape[0])
    model = PopulationComplexMLP(pop, 2 * n, [int(E.shape[2])], int(ckpt["W_out"].shape[-1]), device=device)
    model.E.copy_(E.to(device))
    model.W_out.copy_(ckpt["W_out"].to(device))
    model.b_out.copy_(ckpt["b_out"].to(device))
    task = ModularMultiplicationStar(p=p)
    _, _, x, y = task.full_domain(device)
    ident_acc = (model(x).argmax(dim=-1) == y.unsqueeze(1)).float().mean(dim=0)
    dlog = discrete_log_table(p)
    fits = []
    hom = []
    radii_cv = []
    for i in range(pop):
        xy = E[i, :, 0, :].cpu().numpy().astype(np.float64)
        fit = fit_dlog_circle(xy, dlog, n)
        fits.append(fit)
        hom.append(homomorphism_errors(xy, n))
        _, rad = _angles_and_radii(xy)
        radii_cv.append(float(rad.std() / (rad.mean() + 1e-12)))
    r2 = np.array([f["r2"] for f in fits])
    ks = np.array([f["k"] for f in fits])
    sub = substitute_fitted_circle(model, fits, x, y)
    k_counts = {int(k): int((ks == k).sum()) for k in sorted(set(ks.tolist()))}
    return {
        "n_loaded": pop,
        "identity_n_exact": int((ident_acc >= 1.0 - 1e-12).sum()),
        "mean_circle_r2": float(r2.mean()),
        "median_circle_r2": float(np.median(r2)),
        "min_circle_r2": float(r2.min()),
        "frac_r2_ge_0.85": float((r2 >= 0.85).mean()),
        "frac_r2_ge_0.95": float((r2 >= 0.95).mean()),
        "k_counts": k_counts,
        "mean_radius_cv": float(np.mean(radii_cv)),
        "mean_hom_product_abs_err": float(np.mean([h["mean_product_abs_err"] for h in hom])),
        "mean_hom_angle_err_rad": float(np.mean([h["mean_angle_err_rad"] for h in hom])),
        "median_hom_angle_err_rad": float(np.median([h["median_angle_err_rad"] for h in hom])),
        "substitute_fitted_circle": sub,
    }
