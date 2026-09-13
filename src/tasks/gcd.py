"""GCD meet table and a frequency-preserving destroyed-law control.

Values are 1..N. One-hots and labels are indexed 0..N-1 via value-1.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd

import numpy as np
import torch

from src.tasks.base import SplitKind, TaskSplit

DESTROY_SEED = 20260910


def primes_upto(n: int) -> tuple[int, ...]:
    if n < 2:
        return ()
    s = [True] * (n + 1)
    s[0] = s[1] = False
    for p in range(2, int(n**0.5) + 1):
        if s[p]:
            s[p * p :: p] = [False] * len(s[p * p :: p])
    return tuple(i for i in range(2, n + 1) if s[i])


def gcd_table(n: int) -> np.ndarray:
    t = np.empty((n, n), dtype=np.int64)
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            t[i - 1, j - 1] = gcd(i, j)
    return t


def destroyed_gcd_table(n: int, seed: int = DESTROY_SEED) -> np.ndarray:
    g = gcd_table(n)
    rng = np.random.default_rng(int(seed))
    for _ in range(1000):
        t = np.zeros((n, n), dtype=np.int64)
        off = g[np.triu_indices(n, 1)].copy()
        rng.shuffle(off)
        t[np.triu_indices(n, 1)] = off
        t = t + t.T
        np.fill_diagonal(t, np.diag(g))
        if not np.array_equal(t, g):
            return t
    raise RuntimeError("destroyed table collided with gcd_table")


@dataclass
class IntegerCayleyTask:
    n: int
    name: str
    table: np.ndarray

    def __post_init__(self) -> None:
        if self.n < 2:
            raise ValueError(f"n must be >= 2, got {self.n}")
        t = np.asarray(self.table, dtype=np.int64)
        if t.shape != (self.n, self.n):
            raise ValueError(f"table shape {t.shape} != ({self.n}, {self.n})")
        if t.min() < 1 or t.max() > self.n:
            raise ValueError("table entries must lie in 1..n")
        self.table = t

    @property
    def p(self) -> int:
        return self.n

    @property
    def input_dim(self) -> int:
        return 2 * self.n

    @property
    def output_dim(self) -> int:
        return self.n

    @property
    def n_domain(self) -> int:
        return self.n * self.n

    def encode_inputs(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        n = self.n
        return torch.cat(
            [
                torch.nn.functional.one_hot((a.long() - 1), num_classes=n).float(),
                torch.nn.functional.one_hot((b.long() - 1), num_classes=n).float(),
            ],
            dim=-1,
        )

    def operate(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        ai = (a.long() - 1).detach().cpu().numpy()
        bi = (b.long() - 1).detach().cpu().numpy()
        y = self.table[ai, bi]
        return torch.as_tensor(y, device=a.device, dtype=torch.long)

    def labels(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (self.operate(a, b) - 1).long()

    def full_domain(
        self, device: torch.device | str = "cpu"
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        vals = torch.arange(1, self.n + 1, device=device)
        a = vals.repeat_interleave(self.n)
        b = vals.repeat(self.n)
        return a, b, self.encode_inputs(a, b), self.labels(a, b)

    def split(self, kind: SplitKind | str, seed: int = 0, **kwargs) -> TaskSplit:
        kind = SplitKind(kind)
        n = self.n_domain
        vals = torch.arange(1, self.n + 1)
        a = vals.repeat_interleave(self.n)
        b = vals.repeat(self.n)
        idx = torch.arange(n)

        if kind is SplitKind.RANDOM:
            train_frac = float(kwargs.get("train_frac", 0.7))
            g = torch.Generator().manual_seed(int(seed))
            perm = idx[torch.randperm(n, generator=g)]
            n_train = max(1, min(n - 1, int(round(train_frac * n))))
            train_idx, test_idx = perm[:n_train], perm[n_train:]
            info = {"train_frac": train_frac}

        elif kind is SplitKind.HELD_ROW or kind is SplitKind.HELD_COL:
            held = set(int(v) for v in kwargs.get("held_operands", (self.n,)))
            if any(h < 1 or h > self.n for h in held):
                raise ValueError(f"held values must lie in 1..{self.n}, got {sorted(held)}")
            axis = a if kind is SplitKind.HELD_ROW else b
            mask = torch.tensor([int(v) in held for v in axis.tolist()])
            test_idx, train_idx = idx[mask], idx[~mask]
            train_y = self.labels(a[train_idx], b[train_idx])
            test_y = self.labels(a[test_idx], b[test_idx])
            info = {
                "held_operands": sorted(held),
                "axis": "a" if kind is SplitKind.HELD_ROW else "b",
                "n_train": int((~mask).sum()),
                "n_test": int(mask.sum()),
                "n_train_classes": int(torch.unique(train_y).numel()),
                "n_test_classes": int(torch.unique(test_y).numel()),
            }

        else:
            raise ValueError(f"{kind.value} is not implemented for gcd tables")

        if len(train_idx) == 0 or len(test_idx) == 0:
            raise ValueError(f"{kind.value} produced an empty split for n={self.n}")
        return TaskSplit(train_idx=train_idx.long(), test_idx=test_idx.long(), kind=kind, info=info)

    def describe(self) -> dict:
        return {
            "name": self.name,
            "n": self.n,
            "p": self.p,
            "domain": "integer_cayley",
            "input_encoding": "concat_onehot_1_to_n",
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "n_domain": self.n_domain,
        }


def gcd_meet(n: int = 12) -> IntegerCayleyTask:
    return IntegerCayleyTask(n=n, name="gcd_meet", table=gcd_table(n))


def gcd_destroyed(n: int = 12, seed: int = DESTROY_SEED) -> IntegerCayleyTask:
    return IntegerCayleyTask(n=n, name="gcd_destroyed", table=destroyed_gcd_table(n, seed))


def held_row_coverage(task: IntegerCayleyTask, a0: int) -> dict:
    a, b, _, y = task.full_domain("cpu")
    split = task.split(SplitKind.HELD_ROW, seed=0, held_operands=[a0])
    train_a, train_b = a[split.train_idx], b[split.train_idx]
    train_y, test_y = y[split.train_idx], y[split.test_idx]
    values = set(range(1, task.n + 1))
    return {
        "n": task.n,
        "a0": a0,
        "n_train": int(split.train_idx.numel()),
        "n_test": int(split.test_idx.numel()),
        "n_domain": task.n_domain,
        "expected_train": (task.n - 1) * task.n,
        "expected_test": task.n,
        "a0_appears_as_b_in_train": a0 in set(int(v) for v in train_b.tolist()),
        "a0_never_as_a_in_train": a0 not in set(int(v) for v in train_a.tolist()),
        "n_train_classes": int(train_y.unique().numel()),
        "n_test_classes": int(test_y.unique().numel()),
        "all_classes_in_train": int(train_y.unique().numel()) == int(torch.unique(y).numel()),
        "values_as_b_in_train": values <= set(int(v) for v in train_b.tolist()),
    }
