"""Stage B control-table constructions and structural diagnostics.

Tables are n×n with entries in {0,…,n−1}, n=p−1. Index i is residue i+1.
No training lives here. Discrete logs are unused except Control G’s explicit
product-group encoding (not a dlog of F_p*).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

MASTER_SEED = 20260909
PRIMES = (11, 19, 29)
N_INSTANCES = 5
PRIMARY_POP = 1024
REPL_POP = 256
EPOCHS = 8000
LR = 3e-3
MAX_RETRY = 64
CORRUPT_FRACS = (0.01, 0.05, 0.10, 0.25)
E_INTERCALATE_TARGET = 8

CONTROL_IDS = {
    "genuine": 0,
    "A_isomorphic": 1,
    "B_output_perm": 2,
    "C_slot_relabel": 3,
    "D_random_comm": 4,
    "E_latin_nonassoc": 5,
    "E_latin_comm_nonassoc": 6,
    "F_assoc_damage": 7,
    "G_noncyclic_group": 8,
}


def make_seed(control: str, p: int, instance: int, extra: int = 0, retry: int = 0) -> int:
    cid = CONTROL_IDS[control]
    return int(MASTER_SEED + cid * 1_000_000 + p * 10_000 + instance * 1_000 + extra * 100 + retry)


def genuine_mul(p: int) -> np.ndarray:
    n = p - 1
    a = np.arange(1, p)
    return ((a[:, None] * a[None, :]) % p) - 1


def perm_of_n(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.permutation(n).astype(np.int64)


def inverse_perm(pi: np.ndarray) -> np.ndarray:
    inv = np.empty_like(pi)
    inv[pi] = np.arange(pi.size)
    return inv


def control_A(mul: np.ndarray, pi: np.ndarray) -> np.ndarray:
    """a ★ b = π⁻¹(π(a) * π(b)). Isomorphic cyclic group."""
    inv = inverse_perm(pi)
    return inv[mul[np.ix_(pi, pi)]]


def control_B(mul: np.ndarray, pi: np.ndarray) -> np.ndarray:
    """Output-label permutation: y = π(a*b)."""
    return pi[mul]


def control_C(mul: np.ndarray, pi_a: np.ndarray, pi_b: np.ndarray) -> np.ndarray:
    """Independent slot relabeling: y = πA(a) * πB(b)."""
    return mul[np.ix_(pi_a, pi_b)]


def control_D_from_genuine(mul: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random commutative table with the same output multiset as genuine mul."""
    n = mul.shape[0]
    T = mul.copy()
    upper = [(i, j) for i in range(n) for j in range(i + 1, n)]
    vals = np.array([T[i, j] for i, j in upper], dtype=np.int64)
    rng.shuffle(vals)
    for k, (i, j) in enumerate(upper):
        T[i, j] = T[j, i] = vals[k]
    diag = T.diagonal().copy()
    rng.shuffle(diag)
    np.fill_diagonal(T, diag)
    return T


def _intercalates(T: np.ndarray) -> list[tuple[int, int, int, int]]:
    n = T.shape[0]
    out = []
    for r1 in range(n):
        for r2 in range(r1 + 1, n):
            for c1 in range(n):
                for c2 in range(c1 + 1, n):
                    a, b = int(T[r1, c1]), int(T[r1, c2])
                    c, d = int(T[r2, c1]), int(T[r2, c2])
                    if a == d and b == c and a != b:
                        out.append((r1, r2, c1, c2))
    return out


def _swap_intercalate(T: np.ndarray, rec: tuple[int, int, int, int]) -> None:
    r1, r2, c1, c2 = rec
    T[r1, c1], T[r1, c2], T[r2, c1], T[r2, c2] = T[r1, c2], T[r1, c1], T[r2, c2], T[r2, c1]


def control_E_latin(mul: np.ndarray, rng: np.random.Generator, commutative: bool) -> np.ndarray | None:
    """Nonassociative Latin square via intercalate trades on the group table."""
    T = mul.copy()
    if commutative:
        cands = [(r1, r2, c1, c2) for r1, r2, c1, c2 in _intercalates(T) if r1 == c1 and r2 == c2]
    else:
        cands = [c for c in _intercalates(T) if not (c[0] == c[2] and c[1] == c[3])]
    if not cands:
        return None
    n_swap = 0
    cap = max(1, min(E_INTERCALATE_TARGET, len(cands)))
    target = int(rng.integers(1, cap + 1))
    used = set()
    while n_swap < 100:
        pool = [c for c in (cands if n_swap == 0 else _intercalates(T)) if c not in used]
        if commutative:
            pool = [c for c in pool if c[0] == c[2] and c[1] == c[3]]
        else:
            pool = [c for c in pool if not (c[0] == c[2] and c[1] == c[3])]
        if not pool:
            break
        rec = pool[int(rng.integers(0, len(pool)))]
        _swap_intercalate(T, rec)
        used.add(rec)
        n_swap += 1
        if n_swap >= target and not is_associative(T):
            return T
    if not is_associative(T) and is_latin(T):
        return T
    return None


def _shuffle_equal_weight_pairs(
    T: np.ndarray,
    pairs: list[tuple[int, int]],
    rng: np.random.Generator,
) -> int:
    """Shuffle values among positions of equal multiplicity. Returns n changed pairs."""
    if len(pairs) < 2:
        return 0
    vals = np.array([T[i, j] for i, j in pairs], dtype=np.int64)
    orig = vals.copy()
    rng.shuffle(vals)
    if np.array_equal(vals, orig):
        distinct = [i for i in range(len(orig)) if orig[i] != orig[0]]
        if not distinct:
            return 0
        vals = orig.copy()
        vals[0], vals[distinct[0]] = vals[distinct[0]], vals[0]
    for v, (i, j) in zip(vals, pairs):
        T[i, j] = int(v)
        T[j, i] = int(v)
    return int(np.sum(vals != orig))


def _force_offdiag_twocycle(T: np.ndarray, rng: np.random.Generator) -> int:
    """Smallest frequency-preserving change: swap two distinct off-diagonal pair values."""
    n = T.shape[0]
    off = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rng.shuffle(off)
    for a in range(len(off)):
        i1, j1 = off[a]
        for b in range(a + 1, len(off)):
            i2, j2 = off[b]
            if int(T[i1, j1]) != int(T[i2, j2]):
                T[i1, j1], T[i2, j2] = int(T[i2, j2]), int(T[i1, j1])
                T[j1, i1], T[j2, i2] = int(T[i1, j1]), int(T[i2, j2])
                return 2
    return 0


def control_F_damage(mul: np.ndarray, rng: np.random.Generator, frac: float) -> np.ndarray:
    """Damage associativity while keeping symmetry and exact class frequencies.

    Diagonal cells (multiplicity 1) and off-diagonal unordered pairs (multiplicity 2)
    are shuffled separately. A singleton selection cannot change the table; in that
    case the construction applies a 2-cycle of off-diagonal pairs — the smallest
    frequency-preserving move.
    """
    n = mul.shape[0]
    T = mul.copy()
    off = [(i, j) for i in range(n) for j in range(i + 1, n)]
    diag = [(i, i) for i in range(n)]
    n_un = len(off) + len(diag)
    k = max(1, int(round(frac * n_un)))
    k = min(k, n_un)
    labels = [(0, t) for t in range(len(off))] + [(1, t) for t in range(len(diag))]
    chosen = rng.choice(len(labels), size=k, replace=False)
    sel_off = [off[labels[int(t)][1]] for t in chosen if labels[int(t)][0] == 0]
    sel_diag = [diag[labels[int(t)][1]] for t in chosen if labels[int(t)][0] == 1]
    n_changed = _shuffle_equal_weight_pairs(T, sel_off, rng)
    n_changed += _shuffle_equal_weight_pairs(T, sel_diag, rng)
    if n_changed == 0:
        _force_offdiag_twocycle(T, rng)
    return T


def control_G_product(n: int, a: int, b: int) -> np.ndarray:
    """Cayley table of C_a × C_b (additive), n = a*b."""
    if a * b != n:
        raise ValueError(f"a*b={a * b} != n={n}")
    T = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        x1, y1 = i % a, i // a
        for j in range(n):
            x2, y2 = j % a, j // a
            T[i, j] = ((x1 + x2) % a) + a * ((y1 + y2) % b)
    return T


def g_factors(n: int) -> tuple[int, int] | None:
    """A noncyclic C_a × C_b of order n, or None if only cyclic abelian groups exist."""
    if n == 18:
        return (3, 6)
    if n == 28:
        return (2, 14)
    return None


def is_commutative(T: np.ndarray) -> bool:
    return bool(np.array_equal(T, T.T))


def is_latin(T: np.ndarray) -> bool:
    n = T.shape[0]
    want = np.arange(n)
    for i in range(n):
        if not np.array_equal(np.sort(T[i]), want):
            return False
        if not np.array_equal(np.sort(T[:, i]), want):
            return False
    return True


def associativity_fraction(T: np.ndarray) -> float:
    n = T.shape[0]
    ab = T[:, :, None]
    left = T[ab, np.arange(n)]
    right = T[:, T]
    return float((left == right).mean())


def is_associative(T: np.ndarray) -> bool:
    return associativity_fraction(T) >= 1.0 - 1e-15


def find_identity(T: np.ndarray) -> int | None:
    n = T.shape[0]
    rng = np.arange(n)
    for e in range(n):
        if np.array_equal(T[e], rng) and np.array_equal(T[:, e], rng):
            return e
    return None


def inverses_exist(T: np.ndarray, e: int | None) -> bool:
    if e is None:
        return False
    n = T.shape[0]
    for a in range(n):
        left = np.where(T[a] == e)[0]
        right = np.where(T[:, a] == e)[0]
        if left.size != 1 or right.size != 1 or int(left[0]) != int(right[0]):
            return False
    return True


def is_cyclic_loop(T: np.ndarray, e: int) -> bool:
    n = T.shape[0]
    for g in range(n):
        seen = [e]
        x = g
        ok = True
        for _ in range(n - 1):
            if x in seen:
                ok = False
                break
            seen.append(int(x))
            x = int(T[x, g])
        if ok and x == e and len(set(seen)) == n:
            return True
    return False


def output_entropy(T: np.ndarray) -> float:
    n = T.shape[0]
    _, counts = np.unique(T, return_counts=True)
    p = counts / counts.sum()
    h = float(-(p * np.log(np.maximum(p, 1e-15))).sum())
    return h / math.log(n)


def class_counts(T: np.ndarray) -> list[int]:
    n = T.shape[0]
    return [int((T == c).sum()) for c in range(n)]


def diagnose(T: np.ndarray) -> dict[str, Any]:
    n = T.shape[0]
    A = associativity_fraction(T)
    e = find_identity(T)
    inv = inverses_exist(T, e)
    latin = is_latin(T)
    comm = is_commutative(T)
    assoc = A >= 1.0 - 1e-15
    group = bool(assoc and e is not None and inv)
    cyclic = bool(group and e is not None and is_cyclic_loop(T, e))
    counts = class_counts(T)
    return {
        "n": n,
        "commutative": comm,
        "associative": assoc,
        "associativity_fraction": A,
        "identity": e,
        "identity_exists": e is not None,
        "inverses_exist": inv,
        "latin": latin,
        "is_group": group,
        "is_cyclic_group": cyclic,
        "class_counts": counts,
        "class_counts_min": min(counts),
        "class_counts_max": max(counts),
        "balanced_classes": bool(min(counts) == max(counts) == n),
        "output_entropy_normalized": output_entropy(T),
        "automorphism_count": None,
        "automorphism_note": "not enumerated (n! infeasible)",
    }


def held_row_coverage(T: np.ndarray) -> dict[str, Any]:
    n = T.shape[0]
    a0 = n - 1
    train = T[:a0, :]
    test = T[a0:, :]
    train_classes = set(int(x) for x in np.unique(train))
    test_classes = set(int(x) for x in np.unique(test))
    all_cls = set(range(n))
    return {
        "a0_index": a0,
        "n_train": int(train.size),
        "n_test": int(test.size),
        "a0_never_as_a_in_train": True,
        "a0_appears_as_b_in_train": True,
        "all_classes_in_train": train_classes == all_cls,
        "all_classes_in_test": test_classes == all_cls,
        "n_train_classes": len(train_classes),
        "n_test_classes": len(test_classes),
        "missing_train_classes": sorted(all_cls - train_classes),
        "missing_test_classes": sorted(all_cls - test_classes),
    }


def rng_for(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def job_id(p: int, control: str, instance: int, frac: float | None = None) -> str:
    if frac is None:
        return f"p{p}_{control}_i{instance}"
    return f"p{p}_{control}_f{int(round(100 * frac))}_i{instance}"


def population_for_instance(instance: int) -> int:
    return PRIMARY_POP if instance == 0 else REPL_POP


def planned_jobs() -> list[dict[str, Any]]:
    """Locked training matrix. Do not add cells after seeing results."""
    jobs: list[dict[str, Any]] = []
    five = list(range(N_INSTANCES))
    for p in PRIMES:
        jobs.append(_job(p, "genuine", 0, role="baseline"))
        for inst in five:
            jobs.append(_job(p, "A_isomorphic", inst, role="positive_group_relabel"))
            jobs.append(_job(p, "B_output_perm", inst, role="readout_invariance"))
            jobs.append(_job(p, "C_slot_relabel", inst, role="shared_embedding_break"))
            jobs.append(_job(p, "D_random_comm", inst, role="negative_commutative"))
            jobs.append(_job(p, "E_latin_nonassoc", inst, role="negative_latin"))
            jobs.append(_job(p, "E_latin_comm_nonassoc", inst, role="negative_latin_comm"))
            for k, frac in enumerate(CORRUPT_FRACS):
                jobs.append(
                    _job(p, "F_assoc_damage", inst, role="associativity_dose", frac=frac, extra=k)
                )
        g = g_factors(p - 1)
        if g is None:
            jobs.append(
                _job(
                    p,
                    "G_noncyclic_group",
                    0,
                    role="skipped_no_noncyclic_abelian",
                    skip_reason="only cyclic abelian group of this order (C2×C5 ≅ C10)",
                )
            )
        else:
            jobs.append(_job(p, "G_noncyclic_group", 0, role="noncyclic_abelian", g_factors=g))
    return jobs


def _job(
    p: int,
    control: str,
    instance: int,
    *,
    role: str,
    frac: float | None = None,
    extra: int = 0,
    skip_reason: str | None = None,
    g_factors: tuple[int, int] | None = None,
) -> dict[str, Any]:
    skipped = skip_reason is not None
    return {
        "job_id": job_id(p, control, instance, frac),
        "p": p,
        "n": p - 1,
        "p_mod_4": p % 4,
        "control": control,
        "instance": instance,
        "frac": frac,
        "extra": extra,
        "population": 0 if skipped else population_for_instance(instance),
        "epochs": EPOCHS,
        "lr": LR,
        "optimizer": "adam",
        "init": "xavier_uniform",
        "H": 1,
        "held_symbol_index": p - 2,
        "held_residue_if_genuine": p - 1,
        "role": role,
        "skipped": skipped,
        "skip_reason": skip_reason,
        "g_factors": list(g_factors) if g_factors else None,
        "base_seed": None if skipped else make_seed(control, p, instance, extra, retry=0),
        "train_now": False,
    }


# What each control preserves vs destroys. Used in preregistration and the report.
STRUCTURE_MATRIX = [
    {
        "control": "genuine",
        "law": "group law (unchanged)",
        "preserves": [
            "associativity",
            "commutativity",
            "identity",
            "inverses",
            "Latin square",
            "cyclicity C_{p-1}",
            "numerical residue-product alignment",
        ],
        "destroys": ["nothing (baseline replication)"],
        "kind": "baseline",
    },
    {
        "control": "A_isomorphic",
        "law": "same cyclic group, relabeled",
        "preserves": [
            "associativity",
            "commutativity",
            "identity",
            "inverses",
            "Latin square",
            "abstract cyclicity",
        ],
        "destroys": [
            "numerical alignment of displayed residues with F_p* multiplication",
        ],
        "kind": "positive: scrambled labels, intact law",
    },
    {
        "control": "B_output_perm",
        "law": "deterministic output relabel of genuine products",
        "preserves": [
            "latent genuine products",
            "commutativity of the displayed table",
            "Latin square",
            "class balance",
        ],
        "destroys": [
            "compatibility of output labels with the input group elements",
            "associativity of the displayed magma (generally)",
        ],
        "kind": "readout-invariance",
    },
    {
        "control": "C_slot_relabel",
        "law": "y=πA(a)*πB(b), independent slot perms",
        "preserves": [
            "Latin square",
            "group-derived low-rank product structure",
            "class balance",
        ],
        "destroys": [
            "shared-operand compatibility required by Model C",
            "commutativity (generally, because πA ≠ πB)",
        ],
        "kind": "intermediate: structure without shared-embedding fit",
    },
    {
        "control": "D_random_comm",
        "law": "random symmetric balanced table",
        "preserves": [
            "table size n×n",
            "commutativity",
            "exact output-class frequencies of genuine mul (n per class)",
        ],
        "destroys": [
            "associativity",
            "identity",
            "inverses",
            "cyclicity",
            "Latin square (almost surely)",
        ],
        "kind": "negative: scrambled law, matched commutativity+balance",
    },
    {
        "control": "E_latin_nonassoc",
        "law": "nonassociative Latin square / quasigroup",
        "preserves": [
            "Latin square (row/column permutation property)",
            "class balance",
        ],
        "destroys": [
            "associativity",
            "group axioms",
            "isotopy to C_{p-1} (Albert: nonassociative ⇒ not isotopic to a group)",
        ],
        "kind": "negative: scrambled law, matched Latin structure",
    },
    {
        "control": "E_latin_comm_nonassoc",
        "law": "commutative nonassociative Latin square",
        "preserves": [
            "Latin square",
            "commutativity",
            "class balance",
        ],
        "destroys": ["associativity", "group axioms"],
        "kind": "negative: commutative quasigroup",
    },
    {
        "control": "F_assoc_damage",
        "law": "genuine mul with local symmetric corruptions",
        "preserves": [
            "commutativity",
            "global class frequencies",
            "most of the original table",
        ],
        "destroys": [
            "associativity (dose 1%, 5%, 10%, 25% of unordered entries)",
            "identity / inverses / Latin at sufficient dose",
        ],
        "kind": "intervention: associativity continuum",
    },
    {
        "control": "G_noncyclic_group",
        "law": "C_a × C_b abelian group of order n (when it exists)",
        "preserves": [
            "associativity",
            "commutativity",
            "identity",
            "inverses",
            "Latin square",
            "some finite abelian group law",
        ],
        "destroys": [
            "cyclicity",
            "existence of a faithful 1-D complex character for the whole group",
        ],
        "kind": "positive group / negative for Family F circle",
    },
]


PREDICTIONS = {
    "C1": (
        "Genuine multiplication baselines reproduce previously observed exact "
        "symbolic families under the locked protocol (p=11 both F and Q; "
        "p=19 both F and Q; p=29 F only, Q absent)."
    ),
    "C2": (
        "Abstract group relabeling (Control A) should NOT destroy symbolic "
        "solvability. After undoing the hidden π for analysis, faithful group "
        "characters should be recoverable. Class-2-style decomposition remains "
        "mathematically available for p≡3 (mod 4). This is scrambled labels, "
        "not scrambled law."
    ),
    "C3": (
        "Output-only permutation (Control B) should not necessarily destroy a "
        "latent group representation, because the linear readout can absorb "
        "class relabeling. Output decoding changes; latent Family F/Q may survive."
    ),
    "C4": (
        "Independent A/B slot relabeling (Control C) should strongly suppress "
        "the shared-embedding mechanism and held-row generalization, even if "
        "the table retains group-derived structure."
    ),
    "C5": (
        "Commutativity is insufficient. Random commutative balanced tables "
        "(Control D) should not produce certified Family F or Family Q solvers "
        "at rates comparable to genuine groups."
    ),
    "C6": (
        "Latin / quasigroup structure is insufficient. Nonassociative Latin "
        "squares (Control E) should not reproduce the cyclic-character mechanism "
        "merely because every row and column is a permutation."
    ),
    "C7": (
        "Associativity matters. As Control F progressively damages associativity, "
        "exact symbolic solver frequency should decrease. No specific monotonic "
        "functional form is preregistered."
    ),
    "C8": (
        "Symbolic certification remains strict. No control network counts as "
        "Family F or Q merely because its embedding looks circular or radial. "
        "It must pass functional replacement at 100% full-table accuracy."
    ),
    "C9": (
        "If a destroyed-law control repeatedly yields domain-exact networks with "
        "a compact neural-free symbolic decoder, STOP and classify it as a new "
        "mechanism rather than dismissing it as memorization."
    ),
    "C10": (
        "Failure on a completely random table alone does NOT establish group "
        "discovery. The strongest evidence is the gradient across genuine group, "
        "isomorphic group, output relabeling, structured non-group, Latin/"
        "quasigroup, associativity-damaged, and random commutative tables."
    ),
}


FAITHFUL_CERTIFICATE = [
    "full-domain exactness (float32 domain_acc >= 1.0)",
    "near-constant radius",
    "angular homomorphism after gauge correction",
    "faithful winding",
    "exact ideal-character substitution",
    "neural-free geometric decoder",
    "100% full-table symbolic accuracy",
    "for Control A: analysis in the abstract group coordinates induced by π; then ask whether the net discovered those coordinates without being given π",
]

CLASS2_CERTIFICATE = [
    "full-domain exactness",
    "kernel-2 quotient angular coordinate",
    "{a,-a} quotient fibers in the relevant abstract group",
    "two radial states carrying the C2 coordinate",
    "exact two-level radial replacement",
    "exact quotient-phase replacement",
    "neural-free sector+shell decoder",
    "100% full-table accuracy",
    "equivalence to the CRT program",
    "do not weaken for controls",
]


def _perms_equal(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.array_equal(a, b))


def construct_table(
    control: str,
    p: int,
    instance: int,
    *,
    frac: float | None = None,
    extra: int = 0,
    retry: int = 0,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    mul = genuine_mul(p)
    n = mul.shape[0]
    seed = make_seed(control, p, instance, extra, retry)
    rng = rng_for(seed)
    extras: dict[str, Any] = {"seed": seed, "retry": retry}

    if control == "genuine":
        return mul.copy(), extras

    if control == "A_isomorphic":
        pi = perm_of_n(n, rng)
        extras["pi"] = pi.tolist()
        extras["pi_is_identity"] = bool(np.array_equal(pi, np.arange(n)))
        return control_A(mul, pi), extras

    if control == "B_output_perm":
        pi = perm_of_n(n, rng)
        extras["pi"] = pi.tolist()
        extras["pi_is_identity"] = bool(np.array_equal(pi, np.arange(n)))
        return control_B(mul, pi), extras

    if control == "C_slot_relabel":
        pi_a = perm_of_n(n, rng)
        pi_b = perm_of_n(n, rng)
        extras["pi_a"] = pi_a.tolist()
        extras["pi_b"] = pi_b.tolist()
        extras["pi_a_equals_pi_b"] = _perms_equal(pi_a, pi_b)
        if extras["pi_a_equals_pi_b"]:
            return None, extras
        return control_C(mul, pi_a, pi_b), extras

    if control == "D_random_comm":
        return control_D_from_genuine(mul, rng), extras

    if control == "E_latin_nonassoc":
        T = control_E_latin(mul, rng, commutative=False)
        extras["construction"] = "intercalate_trades_noncomm"
        return T, extras

    if control == "E_latin_comm_nonassoc":
        T = control_E_latin(mul, rng, commutative=True)
        extras["construction"] = "intercalate_trades_comm"
        return T, extras

    if control == "F_assoc_damage":
        if frac is None:
            raise ValueError("F_assoc_damage requires frac")
        extras["frac_target"] = frac
        extras["n_unordered"] = n * (n + 1) // 2
        extras["k_requested"] = max(1, int(round(frac * extras["n_unordered"])))
        extras["singleton_shuffle_is_noop"] = extras["k_requested"] == 1
        extras["construction_note"] = (
            "equal-weight shuffle (diag vs off-diag); 2-cycle fallback if no-op"
        )
        return control_F_damage(mul, rng, frac), extras

    if control == "G_noncyclic_group":
        fac = g_factors(n)
        extras["g_factors"] = list(fac) if fac else None
        if fac is None:
            return None, extras
        return control_G_product(n, fac[0], fac[1]), extras

    raise ValueError(f"unknown control {control!r}")


def intended_structure_ok(
    control: str,
    T: np.ndarray,
    diag: dict[str, Any],
    mul: np.ndarray,
    extras: dict[str, Any],
) -> tuple[bool, str]:
    """Return (ok, reason). Failures trigger the preregistered retry."""
    equal_mul = bool(np.array_equal(T, mul))

    if control == "genuine":
        if not (diag["is_cyclic_group"] and diag["latin"] and equal_mul):
            return False, "genuine table is not the cyclic mul group"
        return True, "ok"

    if control == "A_isomorphic":
        if equal_mul:
            return False, "π is an automorphism; displayed table equals genuine mul"
        if not (diag["is_cyclic_group"] and diag["latin"] and diag["commutative"]):
            return False, "isomorphic relabeling failed to remain a cyclic group"
        return True, "ok"

    if control == "B_output_perm":
        if extras.get("pi_is_identity") or equal_mul:
            return False, "output permutation is identity"
        if not diag["latin"]:
            return False, "output permutation lost Latin structure"
        if not diag["commutative"]:
            return False, "output permutation lost commutativity"
        return True, "ok"

    if control == "C_slot_relabel":
        if extras.get("pi_a_equals_pi_b"):
            return False, "πA equals πB"
        if not diag["latin"]:
            return False, "slot relabeling is not Latin"
        return True, "ok"

    if control == "D_random_comm":
        if not diag["commutative"]:
            return False, "D is not commutative"
        if not diag["balanced_classes"]:
            return False, "D lost class balance"
        if diag["associative"] or diag["is_group"]:
            return False, "D accidentally remained associative/a group"
        return True, "ok"

    if control == "E_latin_nonassoc":
        if T is None:
            return False, "intercalate construction returned None"
        if not diag["latin"]:
            return False, "E is not Latin"
        if diag["associative"] or diag["is_group"]:
            return False, "E remained associative/a group"
        if diag["commutative"]:
            return False, "noncomm E variant is commutative"
        if equal_mul:
            return False, "E equals genuine mul"
        return True, "ok"

    if control == "E_latin_comm_nonassoc":
        if T is None:
            return False, "commutative intercalate construction returned None"
        if not diag["latin"]:
            return False, "E_comm is not Latin"
        if not diag["commutative"]:
            return False, "E_comm is not commutative"
        if diag["associative"] or diag["is_group"]:
            return False, "E_comm remained associative/a group"
        if equal_mul:
            return False, "E_comm equals genuine mul"
        return True, "ok"

    if control == "F_assoc_damage":
        if not diag["commutative"]:
            return False, "F lost commutativity"
        if not diag["balanced_classes"]:
            return False, "F lost global class balance"
        if diag["associative"]:
            return False, "F shuffle did not damage associativity"
        return True, "ok"

    if control == "G_noncyclic_group":
        if extras.get("g_factors") is None:
            return False, "no noncyclic abelian group of this order"
        if not diag["is_group"]:
            return False, "G is not a group"
        if diag["is_cyclic_group"]:
            return False, "G is cyclic"
        if not diag["commutative"]:
            return False, "G is not commutative"
        return True, "ok"

    return False, f"unknown control {control}"


def generate_verified_table(job: dict[str, Any]) -> dict[str, Any]:
    """Generate one control table, retrying until diagnostics and held-row pass."""
    if job.get("skipped"):
        return {
            "job_id": job["job_id"],
            "ok": False,
            "skipped": True,
            "skip_reason": job.get("skip_reason"),
            "control": job["control"],
            "p": job["p"],
            "instance": job["instance"],
        }

    p = int(job["p"])
    control = str(job["control"])
    instance = int(job["instance"])
    frac = job.get("frac")
    extra = int(job.get("extra") or 0)
    mul = genuine_mul(p)
    attempts: list[dict[str, Any]] = []

    for retry in range(MAX_RETRY):
        T, extras = construct_table(control, p, instance, frac=frac, extra=extra, retry=retry)
        if T is None:
            attempts.append({"retry": retry, "reason": "constructor_returned_none", **{k: v for k, v in extras.items() if k != "pi"}})
            continue
        diag = diagnose(T)
        cov = held_row_coverage(T)
        ok, reason = intended_structure_ok(control, T, diag, mul, extras)
        rec = {"retry": retry, "reason": reason, "seed": extras["seed"]}
        attempts.append(rec)
        if not ok:
            continue
        if not cov["all_classes_in_train"]:
            attempts[-1]["reason"] = "held-row train missing an output class"
            continue
        n_diff = int(np.sum(T != mul))
        return {
            "job_id": job["job_id"],
            "ok": True,
            "skipped": False,
            "p": p,
            "n": p - 1,
            "control": control,
            "instance": instance,
            "frac": frac,
            "population": job["population"],
            "seed": extras["seed"],
            "retry": retry,
            "n_retries": retry,
            "diagnostics": diag,
            "held_row": cov,
            "extras": extras,
            "n_entries_differing_from_genuine": n_diff,
            "frac_entries_differing_from_genuine": n_diff / T.size,
            "table": T,
            "attempts_tail": attempts[-5:],
        }

    return {
        "job_id": job["job_id"],
        "ok": False,
        "skipped": False,
        "construction_failed": True,
        "p": p,
        "control": control,
        "instance": instance,
        "frac": frac,
        "max_retry": MAX_RETRY,
        "attempts_tail": attempts[-10:],
        "note": "omitted from training if construction fails; recorded before any training",
    }
