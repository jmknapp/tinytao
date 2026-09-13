"""Torch-free symbolic Model C on F_p*.

An exact H=1 generalizer is the cyclic character
    z(a) = s * exp(i (phi + 2 pi k log_g(a) / (p-1)))
optionally conjugated, composed by complex multiplication, classified
by nearest aligned root. No per-residue residual.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from src.analysis.fourier import discrete_log_table, primitive_root


@dataclass(frozen=True)
class SymbolicFStarMul:
    p: int
    primitive_root: int
    k: int
    scale: float
    phi: float
    conjugate: bool

    def n_group(self) -> int:
        return self.p - 1

    def as_dict(self) -> dict:
        return asdict(self)


def _angles(cert: SymbolicFStarMul, dlog: np.ndarray) -> np.ndarray:
    n = cert.p - 1
    theta = 2.0 * math.pi * cert.k * np.array([int(dlog[a]) for a in range(1, cert.p)], dtype=np.float64) / n
    if cert.conjugate:
        theta = -theta
    return theta


def embeddings(cert: SymbolicFStarMul, dlog: np.ndarray | None = None) -> np.ndarray:
    """Return z(a) as [n, 2] for residues 1..p-1."""
    dlog = discrete_log_table(cert.p) if dlog is None else dlog
    theta = _angles(cert, dlog) + cert.phi
    return cert.scale * np.stack([np.cos(theta), np.sin(theta)], axis=1)


def classify_pairs(cert: SymbolicFStarMul, a: np.ndarray, b: np.ndarray, dlog: np.ndarray | None = None) -> np.ndarray:
    """Return residue classes 0..p-2 for products a*b (a,b in 1..p-1)."""
    dlog = discrete_log_table(cert.p) if dlog is None else dlog
    z = embeddings(cert, dlog)
    za = z[a - 1]
    zb = z[b - 1]
    h = np.stack(
        [
            za[:, 0] * zb[:, 0] - za[:, 1] * zb[:, 1],
            za[:, 0] * zb[:, 1] + za[:, 1] * zb[:, 0],
        ],
        axis=1,
    )
    # W columns = s^2 R_{2 phi} unit_k(c) = product-space image of e(c)
    w = embeddings(cert, dlog)  # same s, phi, k; product of two e(c) is not needed
    # h = s^2 R_{2phi} unit(ab); W[:,c] should be s^2 R_{2phi} unit(c)
    # embeddings(cert) is s R_phi unit(c), NOT s^2 R_{2phi} unit(c).
    # Use doubled phase and squared scale:
    unit_theta = _angles(cert, dlog)
    doubled = (cert.scale**2) * np.stack(
        [np.cos(2.0 * cert.phi + unit_theta), np.sin(2.0 * cert.phi + unit_theta)],
        axis=1,
    )
    logits = h @ doubled.T
    return logits.argmax(axis=1)


def from_fit(p: int, fit: dict) -> SymbolicFStarMul:
    return SymbolicFStarMul(
        p=p,
        primitive_root=int(primitive_root(p)),
        k=int(fit["k"]),
        scale=float(fit["scale"]),
        phi=float(fit["phi"]),
        conjugate=bool(fit["conjugate"]),
    )
