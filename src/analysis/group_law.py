"""Causal group-law tests for Model C 2-D embeddings.

Measurements only. A homomorphism residual is reported both raw
(delta = wrap(theta(ab)-theta(a)-theta(b))) and centered (delta minus
its circular mean). A global phase makes raw delta a nonzero constant
even for an exact representation e(a)=s R exp(i k theta_dlog(a)).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from src.analysis.complex_mech import fit_dlog_circle
from src.analysis.fourier import discrete_log_table, primitive_root
from src.analysis.stats import wilson_interval
from src.population.complex import PopulationComplexMLP
from src.tasks.base import SplitKind
from src.tasks.modular import ModularMultiplicationStar

TWO_PI = 2.0 * math.pi


def wrap(x: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(x), np.cos(x))


def circ_mean(angles: np.ndarray, axis=None) -> np.ndarray:
    return np.arctan2(np.mean(np.sin(angles), axis=axis), np.mean(np.cos(angles), axis=axis))


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


def ab_index_table(p: int) -> np.ndarray:
    n = p - 1
    a = np.arange(1, p)
    ab = (a[:, None] * a[None, :]) % p
    return ab - 1


def homomorphism_delta(xy: np.ndarray, p: int) -> np.ndarray:
    """delta[a_idx, b_idx] = wrap(theta(ab) - theta(a) - theta(b)). xy [n,2]."""
    theta = np.arctan2(xy[:, 1], xy[:, 0])
    ab = ab_index_table(p)
    return wrap(theta[ab] - theta[:, None] - theta[None, :])


def summarize_delta(delta: np.ndarray, train_mask: np.ndarray | None = None) -> dict[str, float]:
    """delta [n,n]. Centered residual is delta minus circular mean over all pairs."""
    mu = float(circ_mean(delta.reshape(-1)))
    centered = wrap(delta - mu)
    abs_c = np.abs(centered)
    abs_raw = np.abs(delta)
    out = {
        "circ_mean_rad": mu,
        "mean_abs_raw_rad": float(abs_raw.mean()),
        "rms_raw_rad": float(np.sqrt((delta**2).mean())),
        "max_abs_raw_rad": float(abs_raw.max()),
        "mean_abs_centered_rad": float(abs_c.mean()),
        "rms_centered_rad": float(np.sqrt((centered**2).mean())),
        "max_abs_centered_rad": float(abs_c.max()),
        "mean_abs_centered_deg": float(np.degrees(abs_c.mean())),
        "max_abs_centered_deg": float(np.degrees(abs_c.max())),
    }
    if train_mask is not None:
        te = ~train_mask
        out["mean_abs_centered_train_rad"] = float(np.abs(centered[train_mask]).mean())
        out["mean_abs_centered_held_rad"] = float(np.abs(centered[te]).mean()) if te.any() else float("nan")
        out["max_abs_centered_held_rad"] = float(np.abs(centered[te]).max()) if te.any() else float("nan")
    return out


def radius_stats(xy: np.ndarray, dlog: np.ndarray, n: int) -> dict[str, float]:
    r = np.linalg.norm(xy, axis=-1)
    residues = np.arange(1, n + 1, dtype=np.float64)
    logs = np.array([int(dlog[i + 1]) for i in range(n)], dtype=np.float64)
    def _corr(x, y) -> float:
        if x.std() < 1e-15 or y.std() < 1e-15:
            return 0.0
        return float(np.corrcoef(x, y)[0, 1])
    return {
        "mean": float(r.mean()),
        "cv": float(r.std() / (r.mean() + 1e-12)),
        "min": float(r.min()),
        "max": float(r.max()),
        "corr_residue": _corr(r, residues),
        "corr_dlog": _corr(r, logs),
    }


def pair_masks(p: int, held_residue: int, axis: str) -> dict[str, np.ndarray]:
    n = p - 1
    a = np.arange(1, p)
    aa, bb = np.meshgrid(a, a, indexing="ij")
    if axis == "a":
        held = aa == held_residue
    else:
        held = bb == held_residue
    return {"held": held, "train": ~held}


@torch.no_grad()
def load_population(ckpt: dict, device: torch.device) -> PopulationComplexMLP:
    E = ckpt["E"]
    pop, n, heads, _ = E.shape
    c = int(ckpt["W_out"].shape[-1])
    model = PopulationComplexMLP(pop, 2 * n, [heads], c, device=device)
    model.E.copy_(E.to(device))
    model.W_out.copy_(ckpt["W_out"].to(device))
    model.b_out.copy_(ckpt["b_out"].to(device))
    return model


@torch.no_grad()
def split_accuracies(
    model: PopulationComplexMLP,
    domain_x: torch.Tensor,
    domain_y: torch.Tensor,
    train_idx: torch.Tensor,
    test_idx: torch.Tensor,
) -> dict[str, torch.Tensor]:
    logits = model(domain_x)
    pred = logits.argmax(dim=-1)
    correct = pred == domain_y.unsqueeze(1)
    return {
        "domain": correct.float().mean(dim=0),
        "train": correct[train_idx].float().mean(dim=0),
        "test": correct[test_idx].float().mean(dim=0),
    }


def exact_counts(acc: torch.Tensor) -> dict[str, float]:
    n = int(acc.numel())
    k = int((acc >= 1.0 - 1e-12).sum())
    p, lo, hi = wilson_interval(k, n)
    return {
        "n": n,
        "n_exact": k,
        "frac_exact": p,
        "frac_exact_lo": lo,
        "frac_exact_hi": hi,
        "mean_acc": float(acc.mean()),
        "min_acc": float(acc.min()),
        "max_acc": float(acc.max()),
    }


def write_exact_embeddings(model: PopulationComplexMLP, fits: list[list[dict]], mode: str) -> None:
    """mode: strict_unit | aligned | fitted."""
    E = model.E
    p = E.shape[1] + 1
    dlog = discrete_log_table(p)
    n = E.shape[1]
    heads = E.shape[2]
    with torch.no_grad():
        for i, net_fits in enumerate(fits):
            for h, fit in enumerate(net_fits):
                unit = exact_unit_circle(int(fit["k"]), dlog, n, bool(fit["conjugate"]))
                if mode == "strict_unit":
                    xy = unit
                elif mode == "aligned":
                    xy = apply_scale_rot(unit, float(fit["scale"]), float(fit["phi"]))
                elif mode == "fitted":
                    xy = fit["fitted"]
                else:
                    raise ValueError(mode)
                E[i, :, h, 0] = torch.as_tensor(xy[:, 0], device=E.device, dtype=E.dtype)
                E[i, :, h, 1] = torch.as_tensor(xy[:, 1], device=E.device, dtype=E.dtype)


def unit_normalize_learned(model: PopulationComplexMLP) -> None:
    with torch.no_grad():
        r = model.E.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        model.E.div_(r)


def rotate_embeddings(model: PopulationComplexMLP, phi: float) -> None:
    R = torch.as_tensor(rot2(phi), device=model.E.device, dtype=model.E.dtype)
    with torch.no_grad():
        model.E.copy_(torch.einsum("pihd,de->pihe", model.E, R.T))


def rotate_readout_for_embedding_rotation(model: PopulationComplexMLP, phi: float) -> None:
    """If e -> R_phi e, hidden product rotates by 2 phi. W_new = R_{2phi}^T W per head."""
    R = torch.as_tensor(rot2(2.0 * phi), device=model.W_out.device, dtype=model.W_out.dtype)
    heads = model.heads
    W = model.W_out.reshape(model.population, heads, 2, model.output_dim)
    with torch.no_grad():
        W.copy_(torch.einsum("de,phec->phdc", R, W))
        model.W_out.copy_(W.reshape(model.population, 2 * heads, model.output_dim))


@torch.no_grad()
def lstsq_readout(model: PopulationComplexMLP, x: torch.Tensor, y: torch.Tensor) -> None:
    h = model.compose(x)
    target = torch.nn.functional.one_hot(y, model.output_dim).float()
    pop = model.population
    for i in range(pop):
        hi = h[:, i]
        A = torch.cat([hi, torch.ones(hi.shape[0], 1, device=hi.device, dtype=hi.dtype)], dim=1)
        sol = torch.linalg.lstsq(A, target).solution
        model.W_out[i].copy_(sol[:-1])
        model.b_out[i].copy_(sol[-1])


def winding_readout(fit: dict, dlog: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Geometric W,b for h = (s R_phi z_a)(s R_phi z_b) = s^2 R_{2phi} z_ab."""
    unit = exact_unit_circle(int(fit["k"]), dlog, n, bool(fit["conjugate"]))
    h_dir = apply_scale_rot(unit, float(fit["scale"]) ** 2, 2.0 * float(fit["phi"]))
    # columns of W are those directions so argmax_c h·W_c picks class c
    W = h_dir.T.copy()
    b = np.zeros(n, dtype=np.float64)
    return W, b


def permute_embedding_by_power(model: PopulationComplexMLP, q: int, p: int) -> None:
    """E_new[a] = E_old[a^q]. Changes winding k -> kq."""
    residues = list(range(1, p))
    src = torch.tensor([pow(a, q, p) - 1 for a in residues], dtype=torch.long, device=model.E.device)
    with torch.no_grad():
        model.E.copy_(model.E[:, src])


def random_permute_embeddings(model: PopulationComplexMLP, seed: int) -> None:
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = model.E.shape[1]
    perm = torch.randperm(n, generator=g)
    with torch.no_grad():
        model.E.copy_(model.E[:, perm.to(model.E.device)])


def perturb_angle(model: PopulationComplexMLP, residue: int, eps: float) -> None:
    """Add eps to arg(e(residue)) on every head."""
    idx = residue - 1
    with torch.no_grad():
        xy = model.E[:, idx]
        r = xy.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        c, s = math.cos(eps), math.sin(eps)
        rot = xy.clone()
        rot[..., 0] = c * xy[..., 0] - s * xy[..., 1]
        rot[..., 1] = s * xy[..., 0] + c * xy[..., 1]
        model.E[:, idx] = rot


def snapshot_params(model: PopulationComplexMLP) -> dict[str, torch.Tensor]:
    return {
        "E": model.E.detach().clone(),
        "W_out": model.W_out.detach().clone(),
        "b_out": model.b_out.detach().clone(),
    }


def restore_params(model: PopulationComplexMLP, snap: dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        model.E.copy_(snap["E"])
        model.W_out.copy_(snap["W_out"])
        model.b_out.copy_(snap["b_out"])


def fit_all_heads(E_cpu: torch.Tensor, dlog: np.ndarray) -> list[list[dict[str, Any]]]:
    pop, n, heads, _ = E_cpu.shape
    out: list[list[dict[str, Any]]] = []
    for i in range(pop):
        row = []
        for h in range(heads):
            xy = E_cpu[i, :, h, :].numpy().astype(np.float64)
            row.append(fit_dlog_circle(xy, dlog, n))
        out.append(row)
    return out


def collect_head0_xy(E_cpu: torch.Tensor) -> np.ndarray:
    return E_cpu[:, :, 0, :].numpy().astype(np.float64)


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
    best_a = np.zeros(pop)
    best_b = np.zeros(pop)
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
            fitted = np.stack([a[:, None] * c - bvec[:, None] * s, bvec[:, None] * c + a[:, None] * s], axis=-1)
            r2 = 1.0 - ((xy - fitted) ** 2).sum(axis=(1, 2)) / y_var
            better = r2 > best_r2
            best_r2 = np.where(better, r2, best_r2)
            best_k = np.where(better, k, best_k)
            best_phi = np.where(better, np.arctan2(bvec, a), best_phi)
            best_scale = np.where(better, np.hypot(a, bvec), best_scale)
            best_conj = np.where(better, conj, best_conj)
            best_a = np.where(better, a, best_a)
            best_b = np.where(better, bvec, best_b)
    return {
        "r2": best_r2,
        "k": best_k,
        "phi": best_phi,
        "scale": best_scale,
        "conjugate": best_conj,
        "a": best_a,
        "b": best_b,
    }


def centered_hom_deg_batch(xy: np.ndarray, p: int) -> np.ndarray:
    """Mean |centered delta| in degrees. xy [P, n, 2]."""
    theta = np.arctan2(xy[:, :, 1], xy[:, :, 0])
    ab = ab_index_table(p)
    delta = wrap(theta[:, ab] - theta[:, :, None] - theta[:, None, :])
    mu = circ_mean(delta.reshape(delta.shape[0], -1), axis=1)
    centered = wrap(delta - mu[:, None, None])
    return np.degrees(np.abs(centered).mean(axis=(1, 2)))


def fits_from_head_batches(batches: list[dict[str, np.ndarray]], n: int, dlog: np.ndarray) -> list[list[dict]]:
    pop = int(batches[0]["k"].shape[0])
    out: list[list[dict]] = []
    for i in range(pop):
        row = []
        for batch in batches:
            unit = exact_unit_circle(int(batch["k"][i]), dlog, n, bool(batch["conjugate"][i]))
            fitted = apply_scale_rot(unit, float(batch["scale"][i]), float(batch["phi"][i]))
            row.append(
                {
                    "r2": float(batch["r2"][i]),
                    "k": int(batch["k"][i]),
                    "phi": float(batch["phi"][i]),
                    "scale": float(batch["scale"][i]),
                    "conjugate": bool(batch["conjugate"][i]),
                    "fitted": fitted,
                }
            )
        out.append(row)
    return out


def batch_fits_as_lists(batch: dict[str, np.ndarray], n: int, dlog: np.ndarray) -> list[list[dict]]:
    return fits_from_head_batches([batch], n, dlog)


def euler_phi(n: int) -> int:
    if n < 1:
        raise ValueError(n)
    return sum(1 for k in range(1, n + 1) if math.gcd(k, n) == 1)


def units_mod(n: int) -> list[int]:
    return [k for k in range(1, n) if math.gcd(k, n) == 1]
