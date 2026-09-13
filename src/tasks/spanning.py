"""Spanning-tree counts on all labeled simple graphs with n vertices.

Domain size 2^{n(n-1)/2}. Labels are Kirchhoff cofactors (integers).
Deletion-contraction is an independent oracle for the same numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import torch

from src.tasks.base import SplitKind, TaskSplit

DESTROY_SEED = 20260910
N_DEFAULT = 5


def edge_list(n: int) -> tuple[tuple[int, int], ...]:
    return tuple((i, j) for i in range(n) for j in range(i + 1, n))


def bits_to_adj(bits: int, n: int, elist: tuple[tuple[int, int], ...]) -> np.ndarray:
    a = np.zeros((n, n), dtype=np.int64)
    for e, (i, j) in enumerate(elist):
        if bits & (1 << e):
            a[i, j] = a[j, i] = 1
    return a


def kirchhoff_tau(adj: np.ndarray) -> int:
    n = adj.shape[0]
    deg = adj.sum(axis=1).astype(np.float64)
    lap = np.diag(deg) - adj.astype(np.float64)
    return int(round(abs(float(np.linalg.det(lap[1:, 1:])))))


def _dc(key: bytes, n: int) -> int:
    adj = np.frombuffer(key, dtype=np.int64).reshape(n, n).copy()
    if n <= 1:
        return 1
    found = None
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] > 0:
                found = (i, j)
                break
        if found:
            break
    if found is None:
        return 0
    i, j = found
    deleted = adj.copy()
    deleted[i, j] -= 1
    deleted[j, i] -= 1
    idx = [v for v in range(n) if v != j]
    contracted = adj[np.ix_(idx, idx)].copy()
    i2 = idx.index(i)
    for v in range(n):
        if v == j or v == i or adj[j, v] == 0:
            continue
        v2 = idx.index(v)
        contracted[i2, v2] += adj[j, v]
        contracted[v2, i2] += adj[j, v]
    return deletion_contraction(deleted) + deletion_contraction(contracted)


def deletion_contraction(adj: np.ndarray) -> int:
    adj = np.asarray(adj, dtype=np.int64)
    return _dc_cached(adj.tobytes(), adj.shape[0])


@lru_cache(maxsize=None)
def _dc_cached(key: bytes, n: int) -> int:
    return _dc(key, n)


def tau_table(n: int) -> np.ndarray:
    elist = edge_list(n)
    n_graphs = 1 << len(elist)
    out = np.empty(n_graphs, dtype=np.int64)
    for g in range(n_graphs):
        out[g] = kirchhoff_tau(bits_to_adj(g, n, elist))
    return out


def destroyed_tau_table(n: int, seed: int = DESTROY_SEED) -> np.ndarray:
    tau = tau_table(n)
    rng = np.random.default_rng(int(seed))
    for _ in range(1000):
        perm = rng.permutation(tau)
        if not np.array_equal(perm, tau):
            return perm
    raise RuntimeError("destroyed τ collided with Kirchhoff")


@dataclass
class SpanningTreeTask:
    n: int = N_DEFAULT
    name: str = "spanning_tau"
    labels_vec: np.ndarray = field(default=None)

    def __post_init__(self) -> None:
        if self.n < 2:
            raise ValueError(f"n must be >= 2, got {self.n}")
        self.elist = edge_list(self.n)
        self.n_edges = len(self.elist)
        self.n_graphs = 1 << self.n_edges
        self.tau_max = self.n ** (self.n - 2)
        if self.labels_vec is None:
            self.labels_vec = tau_table(self.n)
        else:
            self.labels_vec = np.asarray(self.labels_vec, dtype=np.int64)
        if self.labels_vec.shape != (self.n_graphs,):
            raise ValueError("labels_vec length must equal 2^{n_edges}")
        if self.labels_vec.min() < 0 or self.labels_vec.max() > self.tau_max:
            raise ValueError("labels out of range for τ")

    @property
    def p(self) -> int:
        return self.n

    @property
    def input_dim(self) -> int:
        return self.n_edges

    @property
    def output_dim(self) -> int:
        return self.tau_max + 1

    @property
    def n_domain(self) -> int:
        return self.n_graphs

    def encode_inputs(self, bits: torch.Tensor) -> torch.Tensor:
        bits = bits.long()
        e = self.n_edges
        x = torch.zeros(bits.shape[0], e, device=bits.device, dtype=torch.float32)
        for i in range(e):
            x[:, i] = ((bits >> i) & 1).float()
        return x

    def labels(self, bits: torch.Tensor) -> torch.Tensor:
        idx = bits.detach().cpu().numpy().astype(np.int64)
        y = self.labels_vec[idx]
        return torch.as_tensor(y, device=bits.device, dtype=torch.long)

    def full_domain(
        self, device: torch.device | str = "cpu"
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bits = torch.arange(self.n_graphs, device=device)
        dummy = torch.zeros_like(bits)
        return bits, dummy, self.encode_inputs(bits), self.labels(bits)

    def split(self, kind: SplitKind | str, seed: int = 0, **kwargs) -> TaskSplit:
        kind = SplitKind(kind)
        n = self.n_domain
        bits = torch.arange(n)
        idx = torch.arange(n)

        if kind is SplitKind.RANDOM:
            train_frac = float(kwargs.get("train_frac", 0.7))
            g = torch.Generator().manual_seed(int(seed))
            perm = idx[torch.randperm(n, generator=g)]
            n_train = max(1, min(n - 1, int(round(train_frac * n))))
            train_idx, test_idx = perm[:n_train], perm[n_train:]
            info = {"train_frac": train_frac}

        elif kind is SplitKind.HELD_EDGE:
            held = int(kwargs.get("held_operands", [0])[0])
            if held < 0 or held >= self.n_edges:
                raise ValueError(f"held edge must lie in 0..{self.n_edges - 1}")
            mask = ((bits >> held) & 1) == 1
            test_idx, train_idx = idx[mask], idx[~mask]
            train_y = self.labels(bits[train_idx])
            test_y = self.labels(bits[test_idx])
            info = {
                "held_edge": held,
                "held_pair": self.elist[held],
                "n_train": int((~mask).sum()),
                "n_test": int(mask.sum()),
                "n_train_classes": int(torch.unique(train_y).numel()),
                "n_test_classes": int(torch.unique(test_y).numel()),
            }

        else:
            raise ValueError(f"{kind.value} is not implemented for spanning trees")

        if len(train_idx) == 0 or len(test_idx) == 0:
            raise ValueError(f"{kind.value} produced an empty split for n={self.n}")
        return TaskSplit(train_idx=train_idx.long(), test_idx=test_idx.long(), kind=kind, info=info)

    def describe(self) -> dict:
        return {
            "name": self.name,
            "n": self.n,
            "n_edges": self.n_edges,
            "n_graphs": self.n_graphs,
            "tau_max": self.tau_max,
            "input_encoding": "edge_bits",
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "n_domain": self.n_domain,
        }


def spanning_tau(n: int = N_DEFAULT) -> SpanningTreeTask:
    return SpanningTreeTask(n=n, name="spanning_tau")


def spanning_tau_destroyed(n: int = N_DEFAULT, seed: int = DESTROY_SEED) -> SpanningTreeTask:
    return SpanningTreeTask(n=n, name="spanning_tau_destroyed", labels_vec=destroyed_tau_table(n, seed))


def held_edge_coverage(task: SpanningTreeTask, edge: int = 0) -> dict:
    bits, _, _, y = task.full_domain("cpu")
    split = task.split(SplitKind.HELD_EDGE, seed=0, held_operands=[edge])
    all_classes = set(int(v) for v in torch.unique(y).tolist())
    train_classes = set(int(v) for v in torch.unique(y[split.train_idx]).tolist())
    missing = sorted(all_classes - train_classes)
    return {
        "n": task.n,
        "held_edge": edge,
        "held_pair": task.elist[edge],
        "n_train": int(split.train_idx.numel()),
        "n_test": int(split.test_idx.numel()),
        "n_domain": task.n_domain,
        "n_train_classes": len(train_classes),
        "n_all_classes": len(all_classes),
        "missing_classes_in_train": missing,
        "complete_graph_class_test_only": missing == [task.tau_max],
        "train_sees_zero": 0 in train_classes,
    }
