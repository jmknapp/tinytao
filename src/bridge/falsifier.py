"""Attack proposed affine descent lemmas. Exact integer arithmetic, no neural weights."""

from __future__ import annotations

from dataclasses import dataclass

from src.bridge.blackbox import BlackBoxMap
from src.bridge.discoverer import ProposedCertificate, ProposedLemma
from src.bridge.maps import odd_part


@dataclass(frozen=True)
class FalsifyHit:
    lemma: ProposedLemma
    n: int
    reason: str
    observed: int | None


@dataclass(frozen=True)
class FalsifyReport:
    passed: bool
    n_checked: int
    hits: tuple[FalsifyHit, ...]


def _check_lemma(bb: BlackBoxMap, L: ProposedLemma, n: int) -> FalsifyHit | None:
    if n <= 1 or n % 2 == 0:
        return None
    if n % L.M != L.r:
        return None
    if L.kind == "odd_part":
        if L.a is None or L.b is None:
            return FalsifyHit(L, n, "odd_part lemma missing a,b", None)
        try:
            y = bb.F_odd(n)
        except ValueError as e:
            return FalsifyHit(L, n, f"eval error {e}", None)
        pred = odd_part(L.a * n + L.b)
        if y != pred:
            return FalsifyHit(L, n, f"odd_part mismatch pred={pred}", y)
        if y >= n:
            return FalsifyHit(L, n, "no descent", y)
        return None
    q = (n - L.r) // L.M
    try:
        y = bb.F_odd_k(n, L.k)
    except ValueError as e:
        return FalsifyHit(L, n, f"eval error {e}", None)
    pred = L.A * q + L.B
    if y != pred:
        return FalsifyHit(L, n, f"affine mismatch pred={pred}", y)
    if y >= n:
        return FalsifyHit(L, n, "no descent", y)
    return None


def falsify(
    bb: BlackBoxMap,
    cert: ProposedCertificate,
    *,
    n_hi: int = 20_000,
    extra_large: tuple[int, ...] = (10**5 + 1, 10**6 + 17, 10**7 + 3),
) -> FalsifyReport:
    hits: list[FalsifyHit] = []
    checked = 0
    if not cert.lemmas:
        return FalsifyReport(False, 0, (FalsifyHit(ProposedLemma(cert.M, -1, 0, 0, 0, 0), 0, "no lemmas", None),))

    odds = list(range(3, n_hi + 1, 2))
    for n in odds:
        checked += 1
        r = n % cert.M
        L = next((x for x in cert.lemmas if x.r == r), None)
        if L is None:
            hits.append(FalsifyHit(ProposedLemma(cert.M, r, 0, 0, 0, 0), n, "uncovered residue", None))
            if len(hits) >= 8:
                break
            continue
        hit = _check_lemma(bb, L, n)
        if hit is not None:
            hits.append(hit)
            if len(hits) >= 8:
                break

    if not hits:
        for n0 in extra_large:
            n = n0 if n0 % 2 else n0 + 1
            checked += 1
            r = n % cert.M
            L = next((x for x in cert.lemmas if x.r == r), None)
            if L is None:
                hits.append(FalsifyHit(ProposedLemma(cert.M, r, 0, 0, 0, 0), n, "uncovered residue at large n", None))
                continue
            hit = _check_lemma(bb, L, n)
            if hit is not None:
                hits.append(hit)

    return FalsifyReport(passed=not hits, n_checked=checked, hits=tuple(hits[:12]))
