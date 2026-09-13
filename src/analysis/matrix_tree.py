"""Kirchhoff snap certificate for Model K on spanning-tree labels."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from src.population.kirchhoff import laplacian_from_weights
from src.tasks.spanning import (
    DESTROY_SEED,
    N_DEFAULT,
    bits_to_adj,
    deletion_contraction,
    edge_list,
    spanning_tau,
    spanning_tau_destroyed,
    tau_table,
)

N = N_DEFAULT


def adjacency_minor_tau(adj: np.ndarray) -> int:
    a = adj.astype(np.float64)
    return int(round(abs(float(np.linalg.det(a[1:, 1:])))))


def label_oracle_sanity(n: int = N) -> dict[str, Any]:
    elist = edge_list(n)
    n_graphs = 1 << len(elist)
    tau = tau_table(n)
    dc_ok = 0
    adj_eq = 0
    for g in range(n_graphs):
        adj = bits_to_adj(g, n, elist)
        if deletion_contraction(adj) == int(tau[g]):
            dc_ok += 1
        if adjacency_minor_tau(adj) == int(tau[g]):
            adj_eq += 1
    return {
        "n_graphs": n_graphs,
        "dc_equals_kirchhoff": dc_ok == n_graphs,
        "dc_n_correct": dc_ok,
        "adjacency_minor_equals_tau": adj_eq == n_graphs,
        "adjacency_minor_n_correct": adj_eq,
        "tau_max": int(tau.max()),
        "n_unique_tau": int(len(np.unique(tau))),
    }


def _cofactor_from_w(w: np.ndarray, elist: tuple[tuple[int, int], ...], n: int) -> np.ndarray:
    """w [G, E] -> det [G]."""
    wt = torch.as_tensor(w, dtype=torch.float64).unsqueeze(1)
    lap = laplacian_from_weights(wt, elist, n)
    return torch.linalg.det(lap[:, 0, 1:, 1:]).detach().cpu().numpy()


def certify_net(
    scale: np.ndarray,
    bias: np.ndarray,
    x: np.ndarray,
    tau: np.ndarray,
    elist: tuple[tuple[int, int], ...],
    n: int,
    *,
    scramble_seed: int = 0,
) -> dict[str, Any]:
    """x [G, E] in {0,1}, tau [G] integer labels."""
    g, e = x.shape
    out: dict[str, Any] = {
        "certified": False,
        "break_at": None,
        "snap_n_correct": 0,
        "snap_n": g,
        "adj_minor_n_correct": 0,
    }
    w = scale.reshape(1, -1) * x + bias.reshape(1, -1)
    w_bin = (w >= 0.5).astype(np.float64)
    hat = _cofactor_from_w(w_bin, elist, n)
    pred = np.rint(np.abs(hat))
    n_ok = int((pred == tau).sum())
    out["snap_n_correct"] = n_ok
    out["mean_abs_w_minus_x"] = float(np.abs(w - x).mean())
    out["mean_abs_scale"] = float(np.abs(scale).mean())
    out["mean_abs_bias"] = float(np.abs(bias).mean())
    if n_ok != g:
        out["break_at"] = "binarize_weights_not_exact"
        return out
    adj_ok = 0
    for i in range(g):
        bits = 0
        for ee in range(e):
            if x[i, ee] >= 0.5:
                bits |= 1 << ee
        adj = bits_to_adj(bits, n, elist)
        if adjacency_minor_tau(adj) == int(tau[i]):
            adj_ok += 1
    out["adj_minor_n_correct"] = adj_ok
    if adj_ok == g:
        out["break_at"] = "adjacency_minor_still_exact"
        return out
    rng = np.random.default_rng(scramble_seed * 1009 + 7)
    killed = w_bin.copy()
    killed[:, 0] = rng.integers(0, 2, size=g).astype(np.float64)
    hat_k = _cofactor_from_w(killed, elist, n)
    if int((np.rint(np.abs(hat_k)) == tau).sum()) == g:
        out["break_at"] = "drop_edge_0_still_exact"
        return out
    out["certified"] = True
    return out


def decoder_sanity() -> dict[str, Any]:
    task = spanning_tau(N)
    destroyed = spanning_tau_destroyed(N, DESTROY_SEED)
    oracle = label_oracle_sanity(N)
    bits, _, x, y = task.full_domain("cpu")
    xd = x.numpy().astype(np.float64)
    tau = y.numpy().astype(np.int64)
    tau_d = destroyed.labels(bits).numpy().astype(np.int64)
    scale = np.ones(task.n_edges, dtype=np.float64)
    bias = np.zeros(task.n_edges, dtype=np.float64)
    cert = certify_net(scale, bias, xd, tau, task.elist, N, scramble_seed=1)
    cert_d = certify_net(scale, bias, xd, tau_d, task.elist, N, scramble_seed=1)
    return {
        "passed": bool(
            oracle["dc_equals_kirchhoff"]
            and (not oracle["adjacency_minor_equals_tau"])
            and cert["certified"]
            and (not cert_d["certified"])
        ),
        "oracle": oracle,
        "meet": cert,
        "destroyed": cert_d,
        "destroyed_equals_tau": bool(np.array_equal(tau, tau_d)),
    }
