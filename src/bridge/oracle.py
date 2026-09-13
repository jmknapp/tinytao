"""Exact eventual-descent certificates for BridgeMap (oracle only).

A certificate is a modulus M=2^m together with, for every odd residue r,
a horizon k and an affine map F^k(M q + r) = A q + B such that
    A q + B < M q + r
for every integer q >= 0 with n = M q + r > 1.

Small n that fall outside the affine regime (none, if v2 is locked) are
checked by explicit trajectories. Cycles other than the halt state are errors.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.bridge.maps import Affine, BridgeMap, v2


@dataclass(frozen=True)
class ResidueLemma:
    r: int
    k: int
    affine: Affine
    descends_for_all_q: bool
    note: str


@dataclass(frozen=True)
class DescentCertificate:
    map_name: str
    modulus: int
    halt: int
    lemmas: tuple[ResidueLemma, ...]
    small_checked_upto: int
    verified: bool
    failures: tuple[str, ...]


def _odd_iterate_affine(m: BridgeMap, M: int, r: int, k: int) -> Affine:
    """Compose k odd-to-odd steps starting from residue r mod M."""
    aff = Affine(1, 0)  # identity in q, but we track image as aff(q), residue of image
    # Represent current value as M_cur * q_cur + r_cur, with q_cur = aff.A * q + aff.B_q?
    # Simpler: keep current as Affine in original q, and current residue of the *value*
    # value = A q + B, with A even for M-scaled... After first step, image may not
    # be of the form M q' + r' with the same M if A is not a multiple of M.
    #
    # Track value = A q + B exactly. Next odd rule is chosen from (A q + B) mod 2^m_rule.
    # That modulus condition must be independent of q, otherwise lift M.
    val = Affine(M, r)  # n = M q + r
    for _ in range(k):
        n_mod = _constant_mod(val, m.modulus)
        if n_mod % 2 == 0:
            raise ValueError(f"image became even: {val} ≡ {n_mod} (mod {m.modulus})")
        rule = m.rule_for(n_mod)
        # Apply (a n + b) / 2^v with v = v2(a n + b). Need v independent of q.
        # a (A q + B) + b = (a A) q + (a B + b)
        lin_A = rule.a * val.A
        lin_B = rule.a * val.B + rule.b
        v = _constant_v2(lin_A, lin_B, M_orig=M)
        val = Affine(lin_A >> v, lin_B >> v)
        if val.B % 2 == 0:
            raise ValueError(f"non-odd image {val}")
    return val


def _constant_mod(val: Affine, mod: int) -> int:
    """If A ≡ 0 (mod mod), residue is B mod mod, independent of q."""
    if val.A % mod != 0:
        raise ValueError(
            f"residue of {val} mod {mod} depends on q (A not 0 mod {mod}); lift modulus"
        )
    return val.B % mod


def _constant_v2(lin_A: int, lin_B: int, M_orig: int) -> int:
    """v2(lin_A q + lin_B) independent of q >= 0.

    Sufficient: v2(lin_B) < v2(lin_A) (or lin_A == 0).
    """
    if lin_A == 0:
        return v2(lin_B)
    vB = v2(lin_B)
    vA = v2(lin_A)
    if vB >= vA:
        raise ValueError(
            f"v2({lin_A} q + {lin_B}) depends on q (v2(B)={vB} >= v2(A)={vA})"
        )
    return vB


def _min_v2_over_q(lin_A: int, lin_B: int) -> int:
    """Smallest v2(lin_A q + lin_B) for any integer q >= 0. Largest image."""
    if lin_A == 0:
        return v2(lin_B)
    vA, vB = v2(lin_A), v2(lin_B)
    return min(vA, vB)


def one_step_worst_affine(m: BridgeMap, M: int, r: int) -> Affine:
    """F(Mq+r) using the *smallest* possible valuation (upper bound on F)."""
    val = Affine(M, r)
    n_mod = r % m.modulus
    if n_mod % 2 == 0:
        raise ValueError("even residue")
    rule = m.rule_for(n_mod if n_mod != 0 else r)
    lin_A = rule.a * val.A
    lin_B = rule.a * val.B + rule.b
    v = _min_v2_over_q(lin_A, lin_B)
    out = Affine(lin_A >> v, lin_B >> v)
    return out


def descends_affine(val: Affine, M: int, r: int, halt: int = 1) -> tuple[bool, str]:
    """Whether A q + B < M q + r for all q >= 0 with n = M q + r > halt."""
    A, B = val.A, val.B
    q0 = 0
    while M * q0 + r <= halt:
        q0 += 1
        if q0 > 4:
            return False, "no n>halt in this class"
    n0 = M * q0 + r
    f0 = A * q0 + B
    if A > M:
        return False, f"A={A}>M={M}: expands for large q"
    if A == M:
        ok = B < r or n0 == halt
        return ok, f"A=M, B<r is {B}<{r}"
    # A < M: n - F is eventually increasing, so the first n>halt is the worst.
    if f0 < n0:
        return True, f"A={A}<M={M}: worst n={n0} maps to {f0}"
    if f0 == halt and n0 > halt:
        return False, f"A={A}<M but first n={n0} maps to {f0} not strictly below n"
    return False, f"A={A}<M but first n={n0} maps to {f0} >= n"


def prove_eventual_descent(
    m: BridgeMap,
    *,
    modulus: int | None = None,
    max_horizon: int = 8,
    small_upto: int = 4096,
) -> DescentCertificate:
    M = modulus if modulus is not None else max(m.modulus, 32)
    if M & (M - 1):
        raise ValueError("modulus must be a power of 2")
    failures: list[str] = []
    lemmas: list[ResidueLemma] = []

    for r in range(1, M, 2):
        found: ResidueLemma | None = None
        last_err = ""
        for k in range(1, max_horizon + 1):
            try:
                val = _odd_iterate_affine(m, M, r, k)
            except ValueError as e:
                last_err = str(e)
                continue
            ok, note = descends_affine(val, M, r, halt=m.halt)
            if ok:
                found = ResidueLemma(r=r, k=k, affine=val, descends_for_all_q=True, note=note)
                break
            last_err = note
        if found is None:
            try:
                val = one_step_worst_affine(m, M, r)
                ok, note = descends_affine(val, M, r, halt=m.halt)
                if ok:
                    found = ResidueLemma(
                        r=r,
                        k=1,
                        affine=val,
                        descends_for_all_q=True,
                        note="worst-case v2 (q-dependent valuation); " + note,
                    )
                else:
                    last_err = note
            except ValueError as e:
                last_err = str(e)
        if found is None:
            failures.append(f"r={r} mod {M}: no k<={max_horizon} ({last_err})")
        else:
            lemmas.append(found)

    # Explicit check of small integers, including evens, looking for foreign cycles.
    # Skip a long search if the covering already failed (e.g. negative controls).
    check_upto = small_upto if not failures else min(small_upto, 256)
    max_steps = 800 if not failures else 80
    for n in range(1, check_upto + 1):
        try:
            traj = m.trajectory(n, max_steps=max_steps)
        except RuntimeError as e:
            failures.append(str(e))
            continue
        if m.halt not in traj:
            failures.append(f"n={n}: trajectory {traj[:12]}... never hit halt {m.halt}")
        if n > 1 and not any(x < n for x in traj[1:]):
            failures.append(f"n={n}: no later term < n in {traj[:20]}")

    cert = DescentCertificate(
        map_name=m.name,
        modulus=M,
        halt=m.halt,
        lemmas=tuple(lemmas),
        small_checked_upto=check_upto,
        verified=not failures and len(lemmas) == M // 2,
        failures=tuple(failures),
    )
    return cert


def certificate_to_json(cert: DescentCertificate) -> dict:
    return {
        "map_name": cert.map_name,
        "modulus": cert.modulus,
        "halt": cert.halt,
        "small_checked_upto": cert.small_checked_upto,
        "verified": cert.verified,
        "failures": list(cert.failures),
        "lemmas": [
            {
                "r": L.r,
                "k": L.k,
                "A": L.affine.A,
                "B": L.affine.B,
                "descends_for_all_q": L.descends_for_all_q,
                "note": L.note,
            }
            for L in cert.lemmas
        ],
    }
