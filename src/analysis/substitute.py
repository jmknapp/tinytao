"""Substitute fitted 1-D families into first-layer embeddings.

Causal question: is the dlog sinusoid *used*, or only a high-R²
description of u and v? Replacing a side with its fit kills the
residual on F_p*. If domain accuracy stays 1, the residual was not
load-bearing. If it collapses, the fit was correlational.

Residue 0 is never defined by the dlog family; the original u[0], v[0]
are always kept.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from src.analysis.fourier import (
    discrete_log_table,
    reconstruct_dlog_sinusoid,
    reconstruct_residue_sinusoid,
)
from src.population.mlp import PopulationMLP
from src.tasks.modular import build_task


def snapshot_first_two_layers(model: PopulationMLP) -> dict[str, torch.Tensor]:
    out = {
        "W0": model.weights[0].detach().clone(),
        "W1": model.weights[1].detach().clone(),
    }
    if model.biases is not None:
        out["b0"] = model.biases[0].detach().clone()
        out["b1"] = model.biases[1].detach().clone()
    return out


@torch.no_grad()
def restore_first_two_layers(model: PopulationMLP, snap: dict[str, torch.Tensor]) -> None:
    model.weights[0].copy_(snap["W0"])
    model.weights[1].copy_(snap["W1"])
    if model.biases is not None:
        model.biases[0].copy_(snap["b0"])
        model.biases[1].copy_(snap["b1"])


def project_uv(
    u: torch.Tensor,
    v: torch.Tensor,
    family: str,
    which: str = "high",
    thresh: float = 0.85,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Project embeddings onto a 1-D family.

    ``which``:
        high  — replace a side only if that family's R² ≥ thresh
        all   — replace every side
    """
    u_np = u.detach().cpu().numpy().astype(np.float64)
    v_np = v.detach().cpu().numpy().astype(np.float64)
    p = u_np.shape[-1]
    dlog = discrete_log_table(p)
    u_out = u_np.copy()
    v_out = v_np.copy()
    n_replaced = 0
    n_sides = 0
    r2s: list[float] = []
    replaced_per_net = np.zeros(u_np.shape[0], dtype=np.int64)

    def _one(vec: np.ndarray) -> tuple[np.ndarray, float]:
        if family == "dlog":
            yhat, r2, _k = reconstruct_dlog_sinusoid(vec, dlog)
        elif family == "fourier":
            yhat, r2, _k = reconstruct_residue_sinusoid(vec)
        else:
            raise ValueError(f"unknown family {family}")
        return yhat, float(r2)

    for i in range(u_np.shape[0]):
        for h in range(u_np.shape[1]):
            for arr, out in ((u_np, u_out), (v_np, v_out)):
                yhat, r2 = _one(arr[i, h])
                n_sides += 1
                r2s.append(r2)
                if which == "all" or r2 >= thresh:
                    out[i, h] = yhat
                    n_replaced += 1
                    replaced_per_net[i] += 1

    u_t = torch.tensor(u_out, dtype=u.dtype, device=u.device)
    v_t = torch.tensor(v_out, dtype=v.dtype, device=v.device)
    info = {
        "family": family,
        "which": which,
        "thresh": thresh,
        "n_replaced": n_replaced,
        "n_sides": n_sides,
        "n_noop": int((replaced_per_net == 0).sum()),
        "replaced_per_net": replaced_per_net.tolist(),
        "mean_family_r2": float(np.mean(r2s)) if r2s else None,
    }
    return u_t, v_t, info


@torch.no_grad()
def write_embeddings(model: PopulationMLP, u: torch.Tensor, v: torch.Tensor) -> None:
    """u, v are [P, H, p]. W1 is [P, 2p, H]."""
    p = u.shape[-1]
    model.weights[0][:, :p, :].copy_(u.permute(0, 2, 1))
    model.weights[0][:, p:, :].copy_(v.permute(0, 2, 1))


@torch.no_grad()
def refit_readout_lstsq(
    model: PopulationMLP,
    p: int,
    device: torch.device | str,
    task_name: str = "modular_multiplication",
) -> float:
    """Least-squares W2, b2 from projected hidden units onto one-hot labels.

    Returns mean residual of the fit (not accuracy).
    """
    task = build_task(task_name, p)
    _, _, x, y = task.full_domain(device)
    _logits, acts = model(x, hidden=True)
    h = acts[-1]  # [B, P, H]
    bsz, pop, hidden = h.shape
    ones = torch.ones(bsz, pop, 1, device=h.device, dtype=h.dtype)
    a = torch.cat([h, ones], dim=-1).permute(1, 0, 2).contiguous()  # [P, B, H+1]
    target = F.one_hot(y.long(), num_classes=p).to(dtype=h.dtype)
    target = target.unsqueeze(0).expand(pop, -1, -1).contiguous()
    sol = torch.linalg.lstsq(a, target).solution  # [P, H+1, p]
    model.weights[1].copy_(sol[:, :-1, :])
    if model.biases is not None:
        model.biases[1].copy_(sol[:, -1, :])
    recon = torch.einsum("pbk,pkc->pbc", a, sol)
    resid = (recon - target).square().mean().item()
    return float(resid)


def run_substitution_condition(
    model: PopulationMLP,
    snap: dict[str, torch.Tensor],
    u: torch.Tensor,
    v: torch.Tensor,
    spec: dict,
    p: int,
    device: torch.device | str,
    task_name: str = "modular_multiplication",
) -> dict:
    """One projection / optional readout-refit, then domain accuracy on ``task_name``."""
    from src.analysis.ablation import domain_accuracy

    restore_first_two_layers(model, snap)
    info: dict = {"family": spec["family"], "which": spec["which"], "n_replaced": 0, "n_sides": None, "lstsq_resid": None}
    replaced_per_net = None
    if spec["family"] is not None:
        u_p, v_p, info = project_uv(u, v, family=spec["family"], which=spec["which"])
        write_embeddings(model, u_p, v_p)
        replaced_per_net = torch.tensor(info["replaced_per_net"], device=str(device))
    if spec["refit"]:
        info["lstsq_resid"] = refit_readout_lstsq(model, p, device, task_name=task_name)
    acc = domain_accuracy(model, p, device, task_name=task_name)
    n = model.population
    exact = acc >= 1.0 - 1e-12
    n_exact = int(exact.sum())
    n_noop = int((replaced_per_net == 0).sum()) if replaced_per_net is not None else n
    if replaced_per_net is not None:
        touched = replaced_per_net > 0
        n_touched = int(touched.sum())
        n_exact_touched = int((exact & touched).sum())
        mean_touched = float(acc[touched].mean()) if n_touched else None
    else:
        n_touched = 0
        n_exact_touched = n_exact
        mean_touched = float(acc.mean())
    return {
        "name": spec["name"],
        "n_exact": n_exact,
        "n": n,
        "frac_exact": n_exact / n,
        "n_noop": n_noop,
        "n_touched": n_touched,
        "n_exact_touched": n_exact_touched,
        "mean_domain": float(acc.mean()),
        "mean_domain_touched": mean_touched,
        "min_domain": float(acc.min()),
        "max_domain": float(acc.max()),
        "n_replaced": info.get("n_replaced"),
        "n_sides": info.get("n_sides"),
        "mean_family_r2": info.get("mean_family_r2"),
        "lstsq_resid": info.get("lstsq_resid"),
        "per_net": acc.detach().cpu().tolist() if n <= 16 else None,
    }
