"""Stage A.5 hypotheses and winding-census thresholds.

Locked before inspecting non-unit exact embeddings.
"""

from __future__ import annotations

HYPOTHESES = {
    "H_RADIAL_QUOTIENT": (
        "A non-unit winding k is a non-faithful character of C_(p-1). "
        "Phase represents a quotient; radius may encode the lost within-fiber state, "
        "so z(a) ≈ r(a) exp(i theta_k(a)) still multiplies exactly."
    ),
    "H1": "quotient phase + radial fiber code",
    "H2": "distorted but otherwise continuous 2-D lookup geometry",
    "H3": "finite-table memorization enabled by shared embeddings",
    "H4": "another low-complexity symbolic representation not captured by phase/radius",
    "H5": "numerical/readout artifact rather than meaningful internal structure",
}

# Winding census. Homomorphism error is recorded but does not define the class:
# non-unit exact nets are expected to fail the group law on phase alone.
CENSUS_THRESHOLDS = {
    "r2_clear": 0.85,
    "r2_gap": 0.03,
    "note": (
        "faithful_unit: gcd(k_eff,n)=1 AND R²>=0.85 AND not straddling unit/non-unit within gap. "
        "non_unit: gcd>1, even if constant-radius R² is mediocre (expected under H1). "
        "ambiguous: unit k with R²<0.85, or top-two windings straddle unit vs non-unit within gap. "
        "R² is not a gate for non-unit membership: a variable-radius quotient code cannot "
        "match a constant-radius character tightly."
    ),
}

STAGE_A5_PRIMES = (7, 11)
STAGE_A_RUN = "results/runs/20260909_112501_stage_a_primes"

H_RADIAL_QUOTIENT_V2 = (
    "For p=11 non-unit exact solvers: phase represents C10/{±1} ≅ C5, and "
    "radius rank is log_g(a) mod 2. Quotient phases compose by angular addition; "
    "radial states compose through the same-vs-mixed product shell, which implements "
    "parity XOR. The network may encode C10 as a 5-state quotient × a 2-state binary lift. "
    "This is not automatically the group isomorphism C5 × C2 until the operations on "
    "those coordinates are shown to be the group laws."
)

# Written to disk before any Step-2 intervention is evaluated.
PREREGISTRATION = {
    "1_global_radius_level_swap": (
        "Swap the two parity-class mean radii (r_even <-> r_odd), keep phases and W,b. "
        "Same-vs-mixed is invariant, so the XOR bit survives. Mixed-shell (odd product "
        "parity) |h| is unchanged. The two even-parity shells r_even^2 and r_odd^2 exchange. "
        "Predicted labels: unchanged if W,b does not distinguish the two even shells within "
        "a quotient sector; any output changes confined to even-parity cells. Odd-parity "
        "pair accuracy should remain 1."
    ),
    "2_single_fiber_affected_cells": (
        "Fiber F={u,-u}. Swap only those two radii. Let n_in = 1_{a in F}+1_{b in F}. "
        "Predicted changed cells: n_in==1 (exactly one operand in F). n_in==0: embeddings "
        "of both operands unchanged, so h and the label unchanged. n_in==2: both radial "
        "bits flip, XOR is unchanged; |h|(u,u) and |h|(-u,-u) exchange even shells; mixed "
        "|h|(u,-u) unchanged. Under the clean hypothesis n_in==2 labels are unchanged."
    ),
    "3_single_fiber_new_labels": (
        "n_in==0: new label = original. n_in==1: new label = (-original) mod p = p-original. "
        "n_in==2: new label = original (XOR bit cancelled)."
    ),
    "4_double_swapped_operands": (
        "Both operands in F: two bit flips cancel. Predicted label permutation is the "
        "identity on those four cells (u,u), (u,-u), (-u,u), (-u,-u). Even-shell exchange "
        "on (u,u) and (-u,-u) may still move points between r_u^2 and r_{-u}^2; the clean "
        "hypothesis says W treats those as the same lift, so argmax stays."
    ),
    "5_ideal_nonunit_phase": (
        "Replace theta(a) by phi + 2*pi*k*log_g(a)/10, keep learned radii, freeze W,b. "
        "If phase residual is nonessential, domain exactness is preserved. No extra "
        "rotation of W is applied beyond the fitted phi already in the formula. If this "
        "fails, the residual is load-bearing or W is aligned to the distortion."
    ),
    "6_two_level_radii": (
        "Replace all ten radii by (rho_even, rho_odd) according to log_g(a) mod 2, "
        "rho_* = mean learned radius in that parity class. Keep learned phases, freeze W,b. "
        "Then the same with exact non-unit phases. If both succeed, the embedding is the "
        "four-parameter form (k, phi, rho_even, rho_odd) with k discrete."
    ),
    "7_max_acc_after_radius_removal": (
        "Unit-normalize z(a). Phase identifies only {c,-c}, so a deterministic phase-only "
        "decoder has pair accuracy at most 1/2 and cannot be domain-exact (0/95 exact). "
        "An optimal linear readout on unit-normalized embeddings cannot exceed that bound "
        "if a and -a have the same phase. Predicted: frozen W fails; LS readout pair acc "
        "≈ 0.5, exact count = 0."
    ),
    "8_class2_criterion": (
        "Promote a net to CLASS 2 only if (A) a compact symbolic embedding in the frozen "
        "original W,b stays domain-exact, or preferably (B) symbolic embedding plus a "
        "deterministic geometric decoder reproduces all 100 pairs with no neural W,b. "
        "Do not promote on correlation, clustering, or accuracy drops alone. Population "
        "is CLASS 2 if (B) holds for every p=11 non-unit exact net; otherwise CLASS 1 "
        "with the residual named."
    ),
}
