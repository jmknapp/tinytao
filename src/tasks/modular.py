"""Modular arithmetic tasks on Z/pZ.

The complete domain has size p^2 and is always materialized. That is the
point: every evaluation is exhaustive, so "accuracy" is not a sample
estimate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.tasks.base import SplitKind, TaskSplit


def _one_hot(values: torch.Tensor, p: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(values.long(), num_classes=p).float()


@dataclass
class _ModularBase:
    p: int
    name: str

    def __post_init__(self) -> None:
        if self.p < 2:
            raise ValueError(f"p must be >= 2, got {self.p}")

    @property
    def input_dim(self) -> int:
        return 2 * self.p

    @property
    def output_dim(self) -> int:
        return self.p

    @property
    def n_domain(self) -> int:
        return self.p * self.p

    def encode_inputs(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Concatenated one-hot pair: [onehot(a) || onehot(b)] in R^{2p}."""
        return torch.cat([_one_hot(a, self.p), _one_hot(b, self.p)], dim=-1)

    def operate(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def labels(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.operate(a, b).long()

    def full_domain(
        self, device: torch.device | str = "cpu"
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        a = torch.arange(self.p, device=device).repeat_interleave(self.p)
        b = torch.arange(self.p, device=device).repeat(self.p)
        x = self.encode_inputs(a, b)
        y = self.labels(a, b)
        return a, b, x, y

    def split(self, kind: SplitKind | str, seed: int = 0, **kwargs) -> TaskSplit:
        kind = SplitKind(kind)
        n = self.n_domain
        a = torch.arange(self.p).repeat_interleave(self.p)
        b = torch.arange(self.p).repeat(self.p)
        y = self.labels(a, b)
        idx = torch.arange(n)

        if kind is SplitKind.RANDOM:
            train_frac = float(kwargs.get("train_frac", 0.7))
            g = torch.Generator().manual_seed(int(seed))
            perm = idx[torch.randperm(n, generator=g)]
            n_train = max(1, min(n - 1, int(round(train_frac * n))))
            train_idx, test_idx = perm[:n_train], perm[n_train:]
            info = {"train_frac": train_frac}

        elif kind is SplitKind.HELD_OUT_OPERAND:
            held = set(int(v) for v in kwargs.get("held_operands", (self.p - 1,)))
            mask = torch.tensor([(int(ai) in held or int(bi) in held) for ai, bi in zip(a, b)])
            test_idx, train_idx = idx[mask], idx[~mask]
            info = {"held_operands": sorted(held)}

        elif kind is SplitKind.HELD_OUT_QUADRANT:
            cut = int(kwargs.get("cut", (self.p + 1) // 2))
            train_small = bool(kwargs.get("train_small", True))
            small = (a < cut) & (b < cut)
            train_mask = small if train_small else ~small
            train_idx, test_idx = idx[train_mask], idx[~train_mask]
            info = {"cut": cut, "train_small": train_small}

        elif kind is SplitKind.HELD_OUT_ZERO:
            mask = (a == 0) | (b == 0)
            test_idx, train_idx = idx[mask], idx[~mask]
            info = {"held": "zero_operand"}

        elif kind is SplitKind.HELD_OUT_PRODUCT:
            held = set(int(v) for v in kwargs.get("held_products", (0,)))
            mask = torch.tensor([int(yi) in held for yi in y])
            test_idx, train_idx = idx[mask], idx[~mask]
            info = {"held_products": sorted(held)}

        elif kind is SplitKind.HELD_OUT_DLOG_PARITY:
            from src.analysis.fourier import discrete_log_table, primitive_root

            dlog = discrete_log_table(self.p)
            hold_odd = bool(kwargs.get("hold_odd", True))
            mask_list = []
            for ai in a.tolist():
                lg = int(dlog[int(ai)])
                if lg < 0:
                    mask_list.append(False)  # zero stays in train
                else:
                    is_odd = (lg % 2) == 1
                    mask_list.append(is_odd if hold_odd else (not is_odd))
            mask = torch.tensor(mask_list)
            test_idx, train_idx = idx[mask], idx[~mask]
            info = {
                "hold_odd": hold_odd,
                "primitive_root": primitive_root(self.p),
                "held_operands": sorted({int(ai) for ai, m in zip(a.tolist(), mask_list) if m}),
            }

        elif kind is SplitKind.DLOG_CHECKERBOARD:
            raise ValueError("dlog_checkerboard is defined on F_p* (use modular_multiplication_star)")

        elif kind is SplitKind.HELD_ROW or kind is SplitKind.HELD_COL:
            held = set(int(v) for v in kwargs.get("held_operands", (self.p - 1,)))
            axis = a if kind is SplitKind.HELD_ROW else b
            mask = torch.tensor([int(v) in held for v in axis.tolist()])
            test_idx, train_idx = idx[mask], idx[~mask]
            info = {
                "held_operands": sorted(held),
                "axis": "a" if kind is SplitKind.HELD_ROW else "b",
                "n_train": int((~mask).sum()),
                "n_test": int(mask.sum()),
                "n_train_classes": int(torch.unique(y[train_idx]).numel()),
                "n_test_classes": int(torch.unique(y[test_idx]).numel()),
            }

        else:
            raise ValueError(f"unknown split {kind}")

        if len(train_idx) == 0 or len(test_idx) == 0:
            raise ValueError(f"{kind.value} produced an empty split for p={self.p}")

        return TaskSplit(train_idx=train_idx.long(), test_idx=test_idx.long(), kind=kind, info=info)

    def describe(self) -> dict:
        return {
            "name": self.name,
            "p": self.p,
            "input_encoding": "concat_onehot",
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "n_domain": self.n_domain,
            "output": "class_over_residues",
        }


@dataclass
class ModularMultiplication(_ModularBase):
    """y = (a * b) mod p.

    Chosen as the first organism-task: the domain is finite, the rule is
    rigid, and several genuinely different algorithms are available
    (Fourier/bilinear features, discrete-log then add, table lookup with a
    zero special case). See README for the comparison with other toys.
    """

    p: int = 13
    name: str = "modular_multiplication"

    def operate(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a.long() * b.long()) % self.p


@dataclass
class ModularAddition(_ModularBase):
    """y = (a + b) mod p. Implemented for later comparison, not Phase 1."""

    p: int = 13
    name: str = "modular_addition"

    def operate(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a.long() + b.long()) % self.p


@dataclass
class ModularMultiplicationStar:
    """y = (a * b) mod p on F_p* × F_p* (nonzero residues only).

    Isolates the cyclic multiplicative group. Zero is restored later.
    Residues are stored as 1..p-1. One-hots and labels are indexed 0..p-2
    via value-1, so output class k means residue k+1.
    """

    p: int = 13
    name: str = "modular_multiplication_star"

    def __post_init__(self) -> None:
        if self.p < 3:
            raise ValueError(f"F_p* requires p >= 3, got {self.p}")

    @property
    def n_group(self) -> int:
        return self.p - 1

    @property
    def input_dim(self) -> int:
        return 2 * self.n_group

    @property
    def output_dim(self) -> int:
        return self.n_group

    @property
    def n_domain(self) -> int:
        return self.n_group * self.n_group

    def encode_inputs(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        n = self.n_group
        return torch.cat(
            [
                torch.nn.functional.one_hot((a.long() - 1), num_classes=n).float(),
                torch.nn.functional.one_hot((b.long() - 1), num_classes=n).float(),
            ],
            dim=-1,
        )

    def operate(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a.long() * b.long()) % self.p

    def labels(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        y = self.operate(a, b)
        if (y == 0).any():
            raise ValueError("F_p* multiply produced 0; operands must be nonzero")
        return (y - 1).long()

    def full_domain(
        self, device: torch.device | str = "cpu"
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        vals = torch.arange(1, self.p, device=device)
        a = vals.repeat_interleave(self.n_group)
        b = vals.repeat(self.n_group)
        return a, b, self.encode_inputs(a, b), self.labels(a, b)

    def split(self, kind: SplitKind | str, seed: int = 0, **kwargs) -> TaskSplit:
        kind = SplitKind(kind)
        n = self.n_domain
        vals = torch.arange(1, self.p)
        a = vals.repeat_interleave(self.n_group)
        b = vals.repeat(self.n_group)
        idx = torch.arange(n)

        if kind is SplitKind.RANDOM:
            train_frac = float(kwargs.get("train_frac", 0.7))
            g = torch.Generator().manual_seed(int(seed))
            perm = idx[torch.randperm(n, generator=g)]
            n_train = max(1, min(n - 1, int(round(train_frac * n))))
            train_idx, test_idx = perm[:n_train], perm[n_train:]
            info = {"train_frac": train_frac, "domain": "F_p*"}

        elif kind is SplitKind.DLOG_CHECKERBOARD:
            from src.analysis.fourier import discrete_log_table, primitive_root

            dlog = discrete_log_table(self.p)
            r = torch.tensor([int(dlog[int(ai)]) for ai in a.tolist()])
            s = torch.tensor([int(dlog[int(bi)]) for bi in b.tolist()])
            even = ((r + s) % 2) == 0
            train_idx, test_idx = idx[even], idx[~even]
            info = {
                "rule": "(dlog a + dlog b) even in train",
                "primitive_root": primitive_root(self.p),
                "n_train": int(even.sum()),
                "n_test": int((~even).sum()),
                "domain": "F_p*",
            }

        elif kind is SplitKind.HELD_ROW or kind is SplitKind.HELD_COL:
            held = set(int(v) for v in kwargs.get("held_operands", (self.p - 1,)))
            if any(h < 1 or h >= self.p for h in held):
                raise ValueError(f"held residues must lie in 1..{self.p - 1}, got {sorted(held)}")
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
                "domain": "F_p*",
            }

        else:
            raise ValueError(f"{kind.value} is not implemented for F_p*")

        if len(train_idx) == 0 or len(test_idx) == 0:
            raise ValueError(f"{kind.value} produced an empty F_p* split for p={self.p}")
        return TaskSplit(train_idx=train_idx.long(), test_idx=test_idx.long(), kind=kind, info=info)

    def describe(self) -> dict:
        return {
            "name": self.name,
            "p": self.p,
            "domain": "F_p*",
            "input_encoding": "concat_onehot_nonzero",
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "n_domain": self.n_domain,
            "output": "class_over_nonzero_residues",
        }


@dataclass
class CayleyTableStar:
    """Arbitrary n×n magma on symbols 0..n−1, encoded like F_p*.

    Residues are stored as 1..p−1 with p = n+1 (the original prime label).
    Label k means table entry k; operate returns residue table[a−1,b−1]+1.
    Used by Stage B controls. Does not change Model C.
    """

    table: np.ndarray
    p: int
    name: str = "cayley_table_star"

    def __post_init__(self) -> None:
        T = np.asarray(self.table, dtype=np.int64)
        if T.ndim != 2 or T.shape[0] != T.shape[1]:
            raise ValueError(f"table must be square, got {T.shape}")
        if T.shape[0] != self.p - 1:
            raise ValueError(f"table order {T.shape[0]} != p-1={self.p - 1}")
        self.table = T

    @property
    def n_group(self) -> int:
        return self.p - 1

    @property
    def input_dim(self) -> int:
        return 2 * self.n_group

    @property
    def output_dim(self) -> int:
        return self.n_group

    @property
    def n_domain(self) -> int:
        return self.n_group * self.n_group

    def encode_inputs(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        n = self.n_group
        return torch.cat(
            [
                torch.nn.functional.one_hot((a.long() - 1), num_classes=n).float(),
                torch.nn.functional.one_hot((b.long() - 1), num_classes=n).float(),
            ],
            dim=-1,
        )

    def operate(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        ia = (a.long() - 1).cpu().numpy()
        ib = (b.long() - 1).cpu().numpy()
        out = self.table[ia, ib] + 1
        return torch.as_tensor(out, device=a.device, dtype=torch.long)

    def labels(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        ia = (a.long() - 1).cpu().numpy()
        ib = (b.long() - 1).cpu().numpy()
        out = self.table[ia, ib]
        return torch.as_tensor(out, device=a.device, dtype=torch.long)

    def full_domain(
        self, device: torch.device | str = "cpu"
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        vals = torch.arange(1, self.p, device=device)
        a = vals.repeat_interleave(self.n_group)
        b = vals.repeat(self.n_group)
        T = torch.as_tensor(self.table, device=device, dtype=torch.long)
        y = T[a.long() - 1, b.long() - 1]
        return a, b, self.encode_inputs(a, b), y

    def split(self, kind: SplitKind | str, seed: int = 0, **kwargs) -> TaskSplit:
        kind = SplitKind(kind)
        n = self.n_domain
        vals = torch.arange(1, self.p)
        a = vals.repeat_interleave(self.n_group)
        b = vals.repeat(self.n_group)
        idx = torch.arange(n)
        if kind is SplitKind.HELD_ROW or kind is SplitKind.HELD_COL:
            held = set(int(v) for v in kwargs.get("held_operands", (self.p - 1,)))
            if any(h < 1 or h >= self.p for h in held):
                raise ValueError(f"held residues must lie in 1..{self.p - 1}, got {sorted(held)}")
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
                "domain": "cayley_table",
            }
        else:
            raise ValueError(f"{kind.value} is not implemented for CayleyTableStar")
        if len(train_idx) == 0 or len(test_idx) == 0:
            raise ValueError(f"{kind.value} produced an empty split for p={self.p}")
        return TaskSplit(train_idx=train_idx.long(), test_idx=test_idx.long(), kind=kind, info=info)

    def describe(self) -> dict:
        return {
            "name": self.name,
            "p": self.p,
            "domain": "cayley_table",
            "input_encoding": "concat_onehot_nonzero",
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "n_domain": self.n_domain,
            "output": "class_over_table_symbols",
        }


def build_task(name: str, p: int) -> ModularMultiplication | ModularAddition | ModularMultiplicationStar:
    key = name.lower().replace("-", "_")
    if key in {"modular_multiplication", "modmul", "mul"}:
        return ModularMultiplication(p=p)
    if key in {"modular_addition", "modadd", "add"}:
        return ModularAddition(p=p)
    if key in {"modular_multiplication_star", "modmul_star", "mul_star"}:
        return ModularMultiplicationStar(p=p)
    if key in {"gcd_meet", "gcd"}:
        from src.tasks.gcd import gcd_meet

        return gcd_meet(n=p)
    if key in {"gcd_destroyed"}:
        from src.tasks.gcd import gcd_destroyed

        return gcd_destroyed(n=p)
    if key in {"spanning_tau", "spanning"}:
        from src.tasks.spanning import spanning_tau

        return spanning_tau(n=p)
    if key in {"spanning_tau_destroyed"}:
        from src.tasks.spanning import spanning_tau_destroyed

        return spanning_tau_destroyed(n=p)
    raise ValueError(f"unknown task {name!r}")
