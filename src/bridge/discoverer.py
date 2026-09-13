"""Propose affine residue lemmas from finite odd samples. No oracle certificates."""

from __future__ import annotations

from dataclasses import dataclass, replace

from src.bridge.blackbox import BlackBoxMap
from src.bridge.maps import odd_part


CANDIDATE_MODULI = (
    2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16, 18, 20, 24, 30, 32, 36, 48, 64, 128, 256,
)
MAX_K = 8
# Small odd linear numerators for leftover classes where v2 is not locked.
CANDIDATE_AB = (
    (1, 1),
    (1, 3),
    (3, 1),
    (5, 1),
    (3, 5),
    (5, 3),
    (7, 1),
    (1, 5),
)


@dataclass(frozen=True)
class ProposedLemma:
    M: int
    r: int
    k: int
    A: int
    B: int
    n_train: int
    kind: str = "affine"
    a: int | None = None
    b: int | None = None
    source: str = "unspecified"


@dataclass(frozen=True)
class ProposedCertificate:
    map_name: str
    M: int
    k_max: int
    n_max_train: int
    lemmas: tuple[ProposedLemma, ...]
    uncovered: tuple[int, ...]
    train_cover_frac: float


def _fit_affine(qs: list[int], ys: list[int]) -> tuple[int, int] | None:
    if len(qs) < 3:
        return None
    q0, q1 = qs[0], qs[1]
    if q1 == q0:
        return None
    dy, dq = ys[1] - ys[0], q1 - q0
    if dy % dq != 0:
        return None
    A = dy // dq
    B = ys[0] - A * q0
    if any(A * q + B != y for q, y in zip(qs, ys)):
        return None
    return A, B


def _samples(M: int, r: int, n_max: int) -> list[tuple[int, int]]:
    """(q, n) for odd n = M q + r, 1 < n <= n_max."""
    out = []
    q = 0
    while True:
        n = M * q + r
        if n > n_max:
            break
        if n > 1 and n % 2 == 1:
            out.append((q, n))
        q += 1
    return out


def propose_for_modulus(bb: BlackBoxMap, M: int, n_max: int, k_max: int = MAX_K) -> ProposedCertificate:
    lemmas: list[ProposedLemma] = []
    uncovered: list[int] = []
    residues = range(M) if M % 2 == 1 else range(1, M, 2)
    for r in residues:
        pts = _samples(M, r, n_max)
        found = _try_affine(bb, M, r, pts, k=1)
        if found is None:
            found = _fit_odd_part(bb, M, r, n_max)
        if found is None:
            for k in range(2, k_max + 1):
                found = _try_affine(bb, M, r, pts, k=k)
                if found is not None:
                    break
        if found is None:
            if pts:
                uncovered.append(r)
        else:
            lemmas.append(found)
    n_res = len(list(residues))
    cover = len(lemmas) / max(n_res, 1)
    return ProposedCertificate(
        map_name=bb.name,
        M=M,
        k_max=k_max,
        n_max_train=n_max,
        lemmas=tuple(lemmas),
        uncovered=tuple(uncovered),
        train_cover_frac=cover,
    )


def _try_affine(
    bb: BlackBoxMap,
    M: int,
    r: int,
    pts: list[tuple[int, int]],
    k: int,
) -> ProposedLemma | None:
    if not pts:
        return None
    qs = [q for q, _ in pts]
    try:
        ys = [bb.F_odd_k(n, k) for _, n in pts]
    except ValueError:
        return None
    fit = _fit_affine(qs, ys)
    if fit is None:
        return None
    A, B = fit
    # Constant image on an infinite AP is almost always a small-n collapse, not a lemma.
    if A == 0:
        return None
    if all(A * q + B < n for q, n in pts):
        return ProposedLemma(M=M, r=r, k=k, A=A, B=B, n_train=len(pts), source="exact_affine")
    return None


def _fit_odd_part(bb: BlackBoxMap, M: int, r: int, n_max: int) -> ProposedLemma | None:
    """If affine F^k fails, try F(n) = odd_part(a n + b) with small odd a, b."""
    pts = _samples(M, r, n_max)
    if not pts:
        return None
    pairs = [(n, bb.F_odd(n)) for _, n in pts]
    for a, b in CANDIDATE_AB:
        if any(odd_part(a * n + b) != y for n, y in pairs):
            continue
        if any(y >= n for n, y in pairs):
            continue
        return ProposedLemma(
            M=M,
            r=r,
            k=1,
            A=0,
            B=0,
            n_train=len(pts),
            kind="odd_part",
            a=a,
            b=b,
            source="odd_part",
        )
    return None


def fill_uncovered(bb: BlackBoxMap, cert: ProposedCertificate, n_max: int) -> ProposedCertificate:
    """Falsifier leftover: fit an odd-part rule on each uncovered residue."""
    extra: list[ProposedLemma] = []
    still: list[int] = []
    for r in cert.uncovered:
        found = _fit_odd_part(bb, cert.M, r, n_max)
        if found is None:
            still.append(r)
        else:
            extra.append(found)
    lemmas = cert.lemmas + tuple(extra)
    n_res = (cert.M if cert.M % 2 == 1 else cert.M // 2)
    cover = len(lemmas) / max(n_res, 1)
    return replace(cert, lemmas=lemmas, uncovered=tuple(still), train_cover_frac=cover)


def sweep_moduli(
    bb: BlackBoxMap,
    n_max: int,
    moduli: tuple[int, ...] = CANDIDATE_MODULI,
    k_max: int = MAX_K,
) -> list[ProposedCertificate]:
    ranked = []
    for M in moduli:
        proposed = propose_for_modulus(bb, M, n_max, k_max=k_max)
        ranked.append(fill_uncovered(bb, proposed, n_max))
    ranked.sort(
        key=lambda c: (
            -c.train_cover_frac,
            0 if not c.uncovered else 1,
            c.M,
            max((L.k for L in c.lemmas), default=99),
        )
    )
    return ranked
