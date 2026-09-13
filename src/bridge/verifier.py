"""Independent exact verifier: no PyTorch, no floats, no learned weights."""

from __future__ import annotations

from src.bridge.blackbox import BlackBoxMap
from src.bridge.discoverer import ProposedCertificate
from src.bridge.falsifier import FalsifyReport, falsify
from src.bridge.level5 import prove_discovered


LEVELS = (
    "LEVEL_0_empirical",
    "LEVEL_1_survives_finite_testing",
    "LEVEL_2_compact_symbolic_conjecture",
    "LEVEL_3_all_finite_states_enumerated",
    "LEVEL_4_transitions_verified_exact",
    "LEVEL_5_universal_descent",
)


def label(cert: ProposedCertificate, report: FalsifyReport) -> str:
    if not cert.lemmas:
        return LEVELS[0]
    if cert.uncovered:
        if report.passed:
            return LEVELS[2]
        return LEVELS[1]
    if not report.passed:
        return LEVELS[2]
    # Sample-fitted affines plus a large exact scan are not a proof for every q.
    return LEVELS[4]


def verify(bb: BlackBoxMap, cert: ProposedCertificate, n_hi: int = 20_000) -> dict:
    report = falsify(bb, cert, n_hi=n_hi)
    lvl = label(cert, report)
    universal = None
    if report.passed and not cert.uncovered:
        universal = prove_discovered(cert)
        if universal["proven"]:
            lvl = LEVELS[5]
    return {
        "level": lvl,
        "passed_falsifier": report.passed,
        "n_checked": report.n_checked,
        "n_hits": len(report.hits),
        "hits": [
            {"r": h.lemma.r, "k": h.lemma.k, "n": h.n, "reason": h.reason, "observed": h.observed}
            for h in report.hits
        ],
        "cover_frac": cert.train_cover_frac,
        "uncovered": list(cert.uncovered),
        "M": cert.M,
        "n_lemmas": len(cert.lemmas),
        "universal": universal,
    }
