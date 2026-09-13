"""H_CLASS2_MOD4: prospective Class-2 discovery test.

Predictions P1–P10 and the six-prime list are locked before any new training.
Discrete logs and Legendre symbols are analysis-only.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

import numpy as np

from src.analysis.fourier import discrete_log_table, primitive_root
from src.analysis.group_law import TWO_PI, euler_phi, units_mod, wrap
from src.analysis.radial_causal import (
    apply_exact_phases,
    apply_two_level_radii,
    apply_unit_norm,
    dlog_vals,
    permute_radii,
    polar_parts,
    residue_grid,
    rho_parity,
)
from src.analysis.radial_quotient import fiber_members, n_phases, product_hidden
from src.analysis.stage_a5_hypotheses import CENSUS_THRESHOLDS

H_CLASS2_MOD4 = (
    "For H=1 Model C trained on held-row multiplication in F_p*: kernel-2 "
    "Class-2 quotient+radial-lift attractors should preferentially occur when "
    "p ≡ 3 (mod 4), because only then C_(p-1) ≅ C_((p-1)/2) × C2 supports the "
    "exact p=11-style decomposition. For p ≡ 1 (mod 4) this specific symbolic "
    "mechanism should be absent or strongly suppressed. This is a prediction "
    "about gradient-descent discovery, not about the existence of the algorithm."
)

PREDICTIONS = {
    "P1": (
        "Faithful unit-winding solvers remain possible for every prime "
        "(gcd(k, p-1)=1)."
    ),
    "P2": (
        "p ≡ 3 (mod 4): nonzero kernel-2 exact solvers with gcd(k, p-1)=2. "
        "p ≡ 1 (mod 4): the p=11-style Class-2 mechanism is absent or strongly suppressed."
    ),
    "P3": (
        "For Class-2 candidates at p ≡ 3 (mod 4), phase encodes C_(p-1)/{±1} "
        "with (p-1)/2 states. Product phase identifies {c, −c}. Phase-only "
        "singleton accuracy approaches 1/2 when the quotient is clean."
    ),
    "P4": (
        "Radius organizes by log_g(a) mod 2 up to one global high/low flip. "
        "Within every {a, −a} fiber the two elements occupy opposite radial states."
    ),
    "P5": (
        "Class-2 candidates use gcd(k, p-1)=2. Larger non-faithful kernels are "
        "recorded separately and do not count as confirmation of H_CLASS2_MOD4."
    ),
    "P6": (
        "Class 2 requires exact quotient phases + two parity-dependent radii + "
        "a neural-free quotient+shell decoder at 100% full-domain accuracy. "
        "Domain-exactness, gcd=2, or two radius clusters alone are not enough."
    ),
    "P7": (
        "Unit-normalizing a genuine Class-2 embedding leaves only the quotient. "
        "Even an optimally refit linear readout cannot be domain-exact; "
        "phase-only accuracy ≈ 1/2."
    ),
    "P8": (
        "Parity-preserving radius replacements (two-level snap by dlog parity) "
        "preserve the symbolic Class-2 computation. Parity-destroying "
        "permutations destroy it."
    ),
    "P9": "The Class-2 mechanism still uses one complex coordinate (2 real dims).",
    "P10": (
        "Class-2 frequency need not be constant across primes. The primary "
        "prediction is presence and mechanism, not equal basin size."
    ),
}

CLASS2_CRITERIA = {
    "step1_quotient": "a and −a share an angular fiber; product phase identifies {c, −c}",
    "step2_radial": (
        "radius separates every {a, −a} fiber; high/low matches log_g(a) mod 2 "
        "up to one global flip"
    ),
    "step3_shells": "two-level radii produce three shells; mixed ↔ odd product dlog parity",
    "step4_exact_phase": "theta(a) = phi + 2π k log_g(a)/n with gcd(k, n)=2; learned W,b not required",
    "step5_decoder": (
        "neural-free quotient sector + same/mixed shell reconstructs every product; "
        "require 100% of (p-1)^2 pairs"
    ),
    "crt": "geometric decoder equals CRT solver (u=j mod m, s=j mod 2) on every pair",
    "not_enough": [
        "domain exact",
        "gcd(k, n)=2",
        "two apparent radius clusters",
    ],
}

PROSPECTIVE_PRIMES = (19, 23, 31, 29, 37, 41)
OLD_PRIMES = (7, 11, 13, 17)
POPULATION = 1024
EPOCHS = 8000
PARAMS_FORMULA = "5(p-1)"

# Locked from Stage A.5. Not a Class-2 gate; used only to label category D.
CENSUS = dict(CENSUS_THRESHOLDS)


def factor_int(n: int) -> list[int]:
    factors: list[int] = []
    x = int(n)
    d = 2
    while d * d <= x:
        while x % d == 0:
            factors.append(d)
            x //= d
        d += 1
    if x > 1:
        factors.append(x)
    return factors


def pretty_factor(factors: list[int]) -> str:
    c = Counter(factors)
    parts = []
    for q in sorted(c):
        e = c[q]
        parts.append(str(q) if e == 1 else f"{q}^{e}")
    return " * ".join(parts) if parts else "1"


def gcd_eq_windings(n: int, d: int) -> list[int]:
    return [k for k in range(n) if (math.gcd(k, n) if k else n) == d]


def verify_primitive_root(g: int, p: int) -> dict[str, Any]:
    n = p - 1
    factors = sorted(set(factor_int(n)))
    pow_n = pow(int(g), n, p)
    fails = [f for f in factors if pow(int(g), n // f, p) == 1]
    order_ok = pow_n == 1 and not fails
    order = n if order_ok else None
    if not order_ok:
        x = 1
        order = None
        for k in range(1, n + 1):
            x = (x * int(g)) % p
            if x == 1:
                order = k
                break
    return {
        "g": int(g),
        "p": int(p),
        "pow_g_n_mod_p": pow_n,
        "prime_factors_of_n": factors,
        "g_to_n_over_factor_is_1": fails,
        "order": order,
        "order_equals_p_minus_1": bool(order == n),
    }


def cn_iso_cm_x_c2(p: int) -> bool:
    """C_{p-1} ≅ C_m × C2 with m=(p-1)/2 iff m is odd iff p ≡ 3 (mod 4)."""
    n = p - 1
    m = n // 2
    return math.gcd(m, 2) == 1


def prime_card(p: int) -> dict[str, Any]:
    n = p - 1
    m = n // 2
    g = int(primitive_root(p))
    g_check = verify_primitive_root(g, p)
    units = units_mod(n)
    gcd2 = gcd_eq_windings(n, 2)
    factors = factor_int(n)
    iso = cn_iso_cm_x_c2(p)
    return {
        "p": p,
        "p_mod_4": p % 4,
        "n": n,
        "m": m,
        "p_minus_1_factorization": factors,
        "p_minus_1_factorization_pretty": pretty_factor(factors),
        "phi_n": euler_phi(n),
        "unit_k": units,
        "n_unit_k": len(units),
        "gcd2_k": gcd2,
        "n_gcd2_k": len(gcd2),
        "C_n_iso_C_m_x_C2": iso,
        "algebraic_reason": (
            f"gcd(m,2)={math.gcd(m, 2)}; C_{n} ≅ C_{m} × C2 iff gcd(m,2)=1 "
            f"(equivalently p ≡ 3 (mod 4) for odd primes)"
        ),
        "train_pair_count": (n - 1) * n,
        "held_row_pair_count": n,
        "n_domain": n * n,
        "params_h1": 5 * n,
        "params_formula": PARAMS_FORMULA,
        "held_residue": n,
        "primitive_root": g,
        "primitive_root_verification": g_check,
        "params_formula_holds": (5 * n) == (2 * n + 2 * n + n),
        "fiber_parity_separable": bool(m % 2 == 1),
        "note_fiber_parity": (
            "m odd: dlog(−1)=m is odd, so a and −a have opposite dlog parity "
            "(radius can lift the kernel). m even: a and −a have the same dlog "
            "parity, so log_g(·) mod 2 cannot separate {a, −a}."
        ),
    }


def crt_j(u: int, s: int, m: int) -> int | None:
    """Unique j in 0..2m-1 with j≡u (mod m) and j≡s (mod 2), or None."""
    u = int(u) % m
    s = int(s) % 2
    if math.gcd(m, 2) != 1:
        return None
    t = (s - (u % 2)) % 2
    return int(u + m * t)


def crt_product(ja: int, jb: int, m: int, p: int, g: int) -> int | None:
    n = 2 * m
    ua, sa = ja % m, ja % 2
    ub, sb = jb % m, jb % 2
    u = (ua + ub) % m
    s = sa ^ sb
    j = crt_j(u, s, m)
    if j is None:
        return None
    return pow(int(g), int(j) % n, p)


def symbolic_decode_class2(
    xy: np.ndarray,
    k: int,
    phi: float,
    dlog_v: np.ndarray,
    p: int,
    rho_even: float,
    rho_odd: float,
) -> np.ndarray:
    """Quotient sector + same/mixed shell. Returns residues 1..p-1."""
    n = p - 1
    q = n_phases(int(k), n)
    h, pr, pth = product_hidden(xy)
    reps = np.arange(q, dtype=np.float64)
    ideal = 2.0 * phi + TWO_PI * int(k) * reps / n
    delta = np.arctan2(np.sin(pth[..., None] - ideal), np.cos(pth[..., None] - ideal))
    prod_fiber = np.abs(delta).argmin(axis=-1)
    shells = np.array([rho_even**2, rho_even * rho_odd, rho_odd**2], dtype=np.float64)
    nearest = np.abs(pr[..., None] - shells).argmin(axis=-1)
    mixed = nearest == 1
    want_even = ~mixed
    pred = np.full(pr.shape, -1, dtype=np.int64)
    for f in range(q):
        members = [a for a in range(1, p) if int(dlog_v[a - 1] % q) == f]
        even_m = [a for a in members if int(dlog_v[a - 1] % 2) == 0]
        odd_m = [a for a in members if int(dlog_v[a - 1] % 2) == 1]
        even_a = even_m[0] if even_m else (members[0] if members else -1)
        odd_a = odd_m[0] if odd_m else (members[-1] if members else -1)
        mask = prod_fiber == f
        pred[mask & want_even] = even_a
        pred[mask & ~want_even] = odd_a
    return pred


def crt_table(p: int, g: int, dlog_v: np.ndarray) -> np.ndarray:
    n = p - 1
    m = n // 2
    out = np.full((n, n), -1, dtype=np.int64)
    for ia, a in enumerate(range(1, p)):
        ja = int(dlog_v[ia])
        for ib, b in enumerate(range(1, p)):
            jb = int(dlog_v[ib])
            prod = crt_product(ja, jb, m, p, g)
            out[ia, ib] = -1 if prod is None else int(prod)
    return out


def legendre_symbol(a: int, p: int) -> int:
    r = pow(int(a) % p, (p - 1) // 2, p)
    if r == 0:
        return 0
    if r == 1:
        return 1
    return -1


def certify_class2_net(
    xy: np.ndarray,
    k: int,
    phi: float,
    p: int,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Five-step Class-2 certificate plus CRT. Does not use W,b."""
    n = p - 1
    m = n // 2
    g = int(primitive_root(p))
    dlog = discrete_log_table(p)
    dlog_v = dlog_vals(p)
    s = dlog_v % 2
    aa, bb, target = residue_grid(p)
    r, th = polar_parts(xy)
    gcd_k = math.gcd(int(k), n) if int(k) else n
    out: dict[str, Any] = {
        "k": int(k),
        "gcd_k": int(gcd_k),
        "phi": float(phi),
        "certified": False,
        "break_at": None,
    }
    if gcd_k != 2:
        out["break_at"] = "gcd_k_not_2"
        return out

    fibers = fiber_members(int(k), p)
    fiber_ok = all(len(v) == 2 and (v[0] + v[1]) % p == 0 for v in fibers.values())
    collapse = []
    sep_ok = []
    large_parity = []
    for members in fibers.values():
        i, j = members[0] - 1, members[1] - 1
        collapse.append(float(np.abs(wrap(th[i] - th[j]))))
        sep_ok.append(bool(r[i] != r[j] and max(r[i], r[j]) / (min(r[i], r[j]) + 1e-12) > 1.2))
        large = members[0] if r[i] >= r[j] else members[1]
        large_parity.append(int(dlog[large] % 2))
    mean_collapse_deg = float(np.degrees(np.mean(collapse))) if collapse else float("nan")
    out["mean_a_minus_a_phase_gap_deg"] = mean_collapse_deg
    out["fibers_are_pm_a"] = bool(fiber_ok)
    out["frac_fibers_radius_ratio_gt_1_2"] = float(np.mean(sep_ok)) if sep_ok else 0.0
    if large_parity:
        maj = int(np.round(np.mean(large_parity)))
        out["radial_parity_consistency"] = float(np.mean(np.array(large_parity) == maj))
        out["large_radius_dlog_parity"] = maj
    else:
        out["radial_parity_consistency"] = 0.0
        out["large_radius_dlog_parity"] = None

    rho_e, rho_o = rho_parity(r, s)
    out["rho_even"] = rho_e
    out["rho_odd"] = rho_o
    out["q_ratio"] = float(rho_o / (rho_e + 1e-12))

    xy_two = apply_two_level_radii(xy, s, rho_e, rho_o)
    xy_sym = apply_exact_phases(xy_two, int(k), float(phi), dlog_v, n)
    dec = symbolic_decode_class2(xy_sym, int(k), float(phi), dlog_v, p, rho_e, rho_o)
    n_pairs = n * n
    n_ok = int((dec == target).sum())
    out["decoder_n_correct"] = n_ok
    out["decoder_acc"] = n_ok / n_pairs
    out["decoder_exact"] = bool(n_ok == n_pairs)

    h, pr, pth = product_hidden(xy_sym)
    shells = np.array([rho_e**2, rho_e * rho_o, rho_o**2], dtype=np.float64)
    nearest = np.abs(pr[..., None] - shells).argmin(axis=-1)
    mixed = nearest == 1
    tgt_odd = (dlog_v[target - 1] % 2) == 1
    out["shell_xor_acc"] = float((mixed == tgt_odd).mean())
    q = n_phases(int(k), n)
    reps = np.arange(q, dtype=np.float64)
    ideal = 2.0 * phi + TWO_PI * int(k) * reps / n
    delta = np.arctan2(np.sin(pth[..., None] - ideal), np.cos(pth[..., None] - ideal))
    prod_fiber = np.abs(delta).argmin(axis=-1)
    tgt_fiber = np.mod(dlog_v[target - 1], q)
    out["quotient_phase_acc"] = float((prod_fiber == tgt_fiber).mean())
    even_of_fiber = np.zeros(q, dtype=np.int64)
    for f in range(q):
        members = [a for a in range(1, p) if int(dlog_v[a - 1] % q) == f]
        even_m = [a for a in members if int(dlog_v[a - 1] % 2) == 0]
        even_of_fiber[f] = even_m[0] if even_m else (members[0] if members else -1)
    even_lift = even_of_fiber[prod_fiber]
    out["phase_only_even_lift_acc"] = float((even_lift == target).mean())

    crt = crt_table(p, g, dlog_v)
    crt_defined = bool((crt > 0).all())
    out["crt_defined"] = crt_defined
    out["C_n_iso_C_m_x_C2"] = cn_iso_cm_x_c2(p)
    if crt_defined:
        out["decoder_equals_crt"] = bool((dec == crt).all())
        out["crt_exact"] = bool((crt == target).all())
    else:
        out["decoder_equals_crt"] = False
        out["crt_exact"] = False
        out["crt_undefined_reason"] = "gcd(m,2)≠1 so (j mod m, j mod 2) is not a CRT chart on C_{2m}"

    # Legendre vs radial state (analysis only)
    rad_bit = (r > np.median(r)).astype(np.int64)
    leg = np.array([legendre_symbol(a, p) for a in range(1, p)], dtype=np.int64)
    # Legendre +1 vs −1 ↔ dlog even vs odd for primitive-root g
    out["legendre_equals_dlog_parity_sign"] = bool(
        np.all((leg == 1) == (s == 0)) or np.all((leg == 1) == (s == 1))
    )
    # after global flip, large radius vs QR
    qr = (leg == 1).astype(np.int64)
    match = float((rad_bit == qr).mean())
    out["radius_vs_quadratic_character_acc"] = max(match, 1.0 - match)

    # P8: parity-destroying perm (one draw)
    rng = np.random.default_rng(0) if rng is None else rng
    perm_d = rng.permutation(n)
    xy_d = apply_exact_phases(permute_radii(xy, perm_d), int(k), float(phi), dlog_v, n)
    r_d, _ = polar_parts(xy_d)
    # decoder still uses the original parity-keyed rho; this is the destroying test
    dec_d = symbolic_decode_class2(xy_d, int(k), float(phi), dlog_v, p, rho_e, rho_o)
    out["parity_destroying_decoder_acc"] = float((dec_d == target).mean())
    out["parity_preserving_two_level_decoder_exact"] = out["decoder_exact"]

    # unit-norm decoder (shells collapse)
    xy_u = apply_exact_phases(apply_unit_norm(xy), int(k), float(phi), dlog_v, n)
    dec_u = symbolic_decode_class2(xy_u, int(k), float(phi), dlog_v, p, 1.0, 1.0)
    out["unit_norm_decoder_acc"] = float((dec_u == target).mean())
    out["unit_norm_decoder_exact"] = bool((dec_u == target).all())

    if not fiber_ok:
        out["break_at"] = "fibers_not_pm_a"
    elif out["frac_fibers_radius_ratio_gt_1_2"] < 1.0 - 1e-12:
        out["break_at"] = "radius_does_not_separate_fiber"
    elif out["radial_parity_consistency"] < 1.0 - 1e-12:
        out["break_at"] = "radius_not_global_dlog_parity"
    elif out["shell_xor_acc"] < 1.0 - 1e-12:
        out["break_at"] = "shell_xor_fails"
    elif not out["decoder_exact"]:
        out["break_at"] = "decoder_not_exact"
    elif not out["C_n_iso_C_m_x_C2"]:
        out["break_at"] = "crt_unavailable_p_eq_1_mod_4"
    elif not out["decoder_equals_crt"]:
        out["break_at"] = "decoder_crt_mismatch"
    else:
        out["certified"] = True
        out["break_at"] = None
    return out


def category_from_k(k_eff: int, n: int, r2: float, r2_second: float, second_unit: bool) -> str:
    from src.analysis.radial_quotient import classify_winding

    kind = classify_winding(
        int(k_eff),
        n,
        float(r2),
        float(r2_second),
        bool(second_unit),
        float(CENSUS["r2_clear"]),
        float(CENSUS["r2_gap"]),
    )
    gcd = math.gcd(int(k_eff), n) if int(k_eff) else n
    if kind == "ambiguous":
        return "D_ambiguous"
    if kind == "faithful_unit":
        return "A_faithful_candidate"
    if gcd == 2:
        return "B_class2_candidate"
    return "C_other_nonunit"
