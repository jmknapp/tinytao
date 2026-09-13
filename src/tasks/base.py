"""Shared task interface for enumerable discrete problems."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import torch


class SplitKind(str, Enum):
    """Structured holdouts that distinguish memorization from a learned rule.

    RANDOM
        Uniform sample of input pairs. Tests ordinary interpolation. A network
        can succeed here by smoothing a lookup table; this split alone cannot
        show that an algorithm was learned.

    HELD_OUT_OPERAND
        Remove every pair that uses a reserved residue as either operand.
        Success means the network can complete missing rows/columns of the
        operation table, which a pure memorizer of seen pairs cannot do.

    HELD_OUT_QUADRANT
        Train only on a rectangular block of the (a, b) table (e.g. small
        residues) and test on the complement. Tests whether the rule
        extrapolates across the algebraic domain rather than interpolating
        nearby examples.

    HELD_OUT_ZERO
        Hold out every pair with a==0 or b==0. Modular multiplication has a
        structurally special absorbing element; this split asks whether zero
        is handled by the same rule or as a separately memorized case.

    HELD_OUT_PRODUCT
        Hold out all pairs whose product residue lies in a reserved set.
        Tests whether the network computes the operation or has specialized
        detectors for particular output classes.

    HELD_OUT_DLOG_PARITY
        Hold out every pair whose first operand has odd discrete log
        (zeros stay in train). Aimed at the Phase 5 observation that
        exact-solver embeddings fit characters of (Z/pZ)*. Success would
        mean the character interpolates from even logs to odd logs.
        Failure is what a memorizer of even-power rows must do.

    DLOG_CHECKERBOARD
        On F_p*, train where (dlog a + dlog b) is even and test the
        complement. Every operand appears in train; test products all
        have odd discrete log. Completing a row is not interpolation
        along the residue axis.

    HELD_ROW
        On F_p*, hold out every pair with a reserved first operand.
        The held residue still appears as b, so its embedding is trained.
        All 12 product classes remain in train. Tests whether a shared
        or first-slot map transfers to an unseen left factor.

    HELD_COL
        Symmetric to HELD_ROW, holding the second operand.

    HELD_EDGE
        On labeled graphs, hold out every graph that contains a reserved
        edge. The complementary graphs (edge absent) remain in train.
        Tests whether an edge's contribution to τ(G) transfers to graphs
        that were never shown with that edge present.
    """

    RANDOM = "random"
    HELD_OUT_OPERAND = "held_out_operand"
    HELD_OUT_QUADRANT = "held_out_quadrant"
    HELD_OUT_ZERO = "held_out_zero"
    HELD_OUT_PRODUCT = "held_out_product"
    HELD_OUT_DLOG_PARITY = "held_out_dlog_parity"
    DLOG_CHECKERBOARD = "dlog_checkerboard"
    HELD_ROW = "held_row"
    HELD_COL = "held_col"
    HELD_EDGE = "held_edge"


@dataclass(frozen=True)
class TaskSplit:
    train_idx: torch.Tensor
    test_idx: torch.Tensor
    kind: SplitKind
    info: dict


class ArithmeticTask(Protocol):
    name: str
    p: int
    input_dim: int
    output_dim: int
    n_domain: int

    def encode_inputs(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor: ...

    def labels(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor: ...

    def full_domain(self, device: torch.device | str = "cpu") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (a, b, x, y) covering every input pair exactly once."""

    def split(self, kind: SplitKind, seed: int, **kwargs) -> TaskSplit: ...

    def describe(self) -> dict: ...
