"""Permutation-aware behavioral descriptors.

Hidden units may be reordered. Distances and features here either align
units (Hungarian matching) or sort them so that raw parameter Euclidean
distance is never used as a similarity.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from src.population.mlp import PopulationMLP
from src.tasks.modular import ModularMultiplication


@torch.no_grad()
def hidden_maps(model: PopulationMLP, p: int, device: torch.device | str) -> torch.Tensor:
    """Hidden activations on the full table, shaped [P, H, p, p].

    Indexing is (network, unit, a, b). Domain order matches
    ``ModularMultiplication.full_domain``: a slow, b fast.
    """
    task = ModularMultiplication(p=p)
    _, _, x, _ = task.full_domain(device)
    _, acts = model(x, hidden=True)
    h = acts[-1]  # [p*p, P, H]
    hidden = h.shape[-1]
    return h.permute(1, 2, 0).reshape(model.population, hidden, p, p).contiguous()


@torch.no_grad()
def embeddings(model: PopulationMLP, p: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """First-layer residue embeddings.

    Because the input is concatenated one-hots,
        h_i = relu(u_i[a] + v_i[b] + c_i)
    with u = W1[:p], v = W1[p:].
    Returns u, v, c each [P, H, p] / [P, H].
    """
    W1 = model.weights[0].detach()
    u = W1[:, :p, :].permute(0, 2, 1).contiguous()
    v = W1[:, p:, :].permute(0, 2, 1).contiguous()
    c = model.biases[0].detach() if model.biases is not None else torch.zeros(model.population, u.shape[1], device=W1.device)
    return u, v, c


def _corr_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson correlation of columns. a,b are [N, H]."""
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    a = a / (np.linalg.norm(a, axis=0, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=0, keepdims=True) + 1e-12)
    return a.T @ b


def align_units(maps_i: torch.Tensor, maps_j: torch.Tensor) -> tuple[np.ndarray, float]:
    """Hungarian match of hidden units by |correlation| of flattened maps.

    ``maps_*`` are [H, p, p]. Returns (permutation of j onto i, mean |corr|).
    """
    a = maps_i.reshape(maps_i.shape[0], -1).T.cpu().numpy()
    b = maps_j.reshape(maps_j.shape[0], -1).T.cpu().numpy()
    corr = _corr_matrix(a, b)
    ri, ci = linear_sum_assignment(-np.abs(corr))
    # permutation[k] = index in j that matches unit k in i
    perm = np.empty(maps_i.shape[0], dtype=np.int64)
    perm[ri] = ci
    score = float(np.abs(corr[ri, ci]).mean()) if len(ri) else 0.0
    return perm, score


def pairwise_aligned_distance(maps: torch.Tensor) -> np.ndarray:
    """1 - mean|corr| after alignment. ``maps`` is [P, H, p, p]."""
    n = maps.shape[0]
    d = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            _, score = align_units(maps[i], maps[j])
            d[i, j] = d[j, i] = 1.0 - score
    return d


def invariant_features(maps: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> np.ndarray:
    """Permutation-invariant fingerprint: per-unit spectra, sorted by energy.

    For each unit: rFFT magnitudes of u and v (length p//2+1 each), plus
    sparsity of the 2D map. Units are sorted by map energy so two networks
    that differ by a hidden permutation have the same vector.
    """
    p = maps.shape[-1]
    n_net, h = maps.shape[:2]
    u_np = u.cpu().numpy()
    v_np = v.cpu().numpy()
    maps_np = maps.cpu().numpy()
    nfreq = p // 2 + 1
    feats = []
    for i in range(n_net):
        energy = (maps_np[i] ** 2).sum(axis=(1, 2))
        order = np.argsort(-energy)
        rows = []
        for t in order:
            uf = np.abs(np.fft.rfft(u_np[i, t]))
            vf = np.abs(np.fft.rfft(v_np[i, t]))
            uf = uf / (uf.sum() + 1e-12)
            vf = vf / (vf.sum() + 1e-12)
            sparse = float((maps_np[i, t] <= 0).mean())
            rows.append(np.concatenate([uf[:nfreq], vf[:nfreq], [sparse]]))
        feats.append(np.concatenate(rows))
    return np.stack(feats, axis=0)
