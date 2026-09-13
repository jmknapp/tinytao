"""Upgrade a discovered covering to a universal descent theorem.

Uses the map definition (residue-linear odd step) and exact 2-adic algebra.
Does not search for lemmas. Discoverer must not import this module.
"""

from __future__ import annotations

from math import gcd

from src.bridge.discoverer import ProposedCertificate, ProposedLemma
from src.bridge.maps import Affine, BridgeMap
from src.bridge.oracle import (
    _odd_iterate_affine,
    descends_affine,
    one_step_worst_affine,
)


MAX_LIFT = 4096

THEOREM = (
    "If n is even and n>1 then T(n)=n/2<n. If n is odd and n>1 then some "
    "F^k(n)<n by a covering lemma proved for every q. A strictly decreasing "
    "chain of positive integers is finite, so every trajectory reaches 1. "
    "A cycle of integers all >1 is impossible."
)


def prove_discovered(cert: ProposedCertificate) -> dict:
    from src.bridge.catalog import get_bridge_map

    failures: list[str] = []
    lemma_notes: list[dict] = []
    if cert.uncovered:
        return _result(False, ["uncovered residues"] + [f"r={r}" for r in cert.uncovered], [])
    try:
        m = get_bridge_map(cert.map_name)
    except KeyError as e:
        return _result(False, [f"no map {e}"], [])

    need = set(range(1, cert.M, 2)) if cert.M % 2 == 0 else {r for r in range(cert.M) if r % 2 == 1}
    have = {L.r for L in cert.lemmas}
    if have != need:
        failures.append(f"covering mismatch have={sorted(have)} need={sorted(need)}")

    for L in cert.lemmas:
        ok, note = _prove_lemma(m, L)
        lemma_notes.append({"r": L.r, "kind": L.kind, "k": L.k, "ok": ok, "note": note})
        if not ok:
            failures.append(f"r={L.r} {L.kind}: {note}")

    even_ok, even_note = _even_descent(m)
    if not even_ok:
        failures.append(even_note)

    proven = not failures
    return {
        "proven": proven,
        "level": "LEVEL_5_universal_descent" if proven else "LEVEL_4_not_universal",
        "theorem": THEOREM if proven else None,
        "failures": failures,
        "lemmas": lemma_notes,
        "even_step": even_note,
        "M": cert.M,
        "map": cert.map_name,
    }


def _result(proven: bool, failures: list[str], notes: list[dict]) -> dict:
    return {
        "proven": proven,
        "level": "LEVEL_5_universal_descent" if proven else "LEVEL_4_not_universal",
        "theorem": THEOREM if proven else None,
        "failures": failures,
        "lemmas": notes,
        "even_step": None,
        "M": None,
        "map": None,
    }


def _even_descent(m: BridgeMap) -> tuple[bool, str]:
    if m.T(2) != 1 or m.T(10) != 5:
        return False, "even rule is not n/2"
    return True, "even n>1: T(n)=n/2<n by definition"


def _prove_lemma(m: BridgeMap, L: ProposedLemma) -> tuple[bool, str]:
    if L.kind == "odd_part":
        return _prove_odd_part(m, L)
    if L.kind == "affine":
        return _prove_affine(m, L)
    return False, f"unknown kind {L.kind}"


def _class_rule_residues(M: int, r: int, mod: int) -> list[int]:
    g = gcd(M, mod)
    out = []
    for s in range(1, mod, 2):
        if (r - s) % g == 0:
            out.append(s)
    return out


def _prove_odd_part(m: BridgeMap, L: ProposedLemma) -> tuple[bool, str]:
    if L.a is None or L.b is None:
        return False, "missing a,b"
    if L.k != 1:
        return False, "odd_part lemma must be k=1"
    residues = _class_rule_residues(L.M, L.r, m.modulus)
    if not residues:
        return False, "empty CRT class"
    for s in residues:
        rule = m.rule_for(s)
        if (rule.a, rule.b) != (L.a, L.b):
            return False, f"rule at s={s} is ({rule.a},{rule.b}) not ({L.a},{L.b})"
    # Identity: every n≡r (mod M) is ≡ some s (mod map.modulus) with that same (a,b),
    # so F(n)=odd_part(a n+b) by definition of the map.
    try:
        worst = one_step_worst_affine(m, L.M, L.r)
    except ValueError as e:
        return False, str(e)
    ok, note = descends_affine(worst, L.M, L.r, halt=m.halt)
    if not ok:
        return False, "identity holds but worst-case image does not descend: " + note
    return True, f"F=odd_part({L.a}n+{L.b}) on class; worst-case {worst.A}q+{worst.B}; {note}"


def _claimed_on_lift(L: ProposedLemma, M_lift: int, r_lift: int) -> Affine:
    if M_lift % L.M:
        raise ValueError("lift is not a multiple of M")
    if (r_lift - L.r) % L.M:
        raise ValueError("lift residue not in class")
    alpha = M_lift // L.M
    beta = (r_lift - L.r) // L.M
    return Affine(L.A * alpha, L.A * beta + L.B)


def _prove_affine(m: BridgeMap, L: ProposedLemma) -> tuple[bool, str]:
    if L.M <= 0 or (L.M & (L.M - 1)):
        return False, "affine Level 5 requires power-of-two M"
    ok_desc, desc_note = descends_affine(Affine(L.A, L.B), L.M, L.r, halt=m.halt)
    if not ok_desc:
        return False, "claimed affine does not descend for all q: " + desc_note
    M = L.M
    last = "no lift"
    while M <= MAX_LIFT:
        try:
            for r_lift in range(1, M, 2):
                if r_lift % L.M != L.r:
                    continue
                got = _odd_iterate_affine(m, M, r_lift, L.k)
                claimed = _claimed_on_lift(L, M, r_lift)
                if got != claimed:
                    return False, (
                        f"at lift M={M} r={r_lift}: F^{L.k}={got.A}q+{got.B} "
                        f"but claimed lemma induces {claimed.A}q+{claimed.B}"
                    )
            return True, f"F^{L.k}={L.A}q+{L.B} locked at lift {M}; {desc_note}"
        except ValueError as e:
            last = str(e)
            M *= 2
    return False, f"could not lock 2-adic residue/valuation ({last})"
