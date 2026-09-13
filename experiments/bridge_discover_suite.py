"""Blind Discoverer–Falsifier on the full bridge catalog. Oracle JSON is read only after discovery."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from src.bridge.blackbox import get_blackbox
from src.bridge.discoverer import sweep_moduli
from src.bridge.verifier import verify


MAPS = (
    "trivial_plus_one",
    "easy_half_collatz",
    "medium_five_on_1_mod_16",
    "hard_five_on_1_mod_16",
    "control_five_on_1_mod_4",
)
N_TRAIN = 2_000
N_FALSIFY = 20_000
SEED = 20260909
ORACLE_DIR = Path("tiny_tao_results/bridge_oracle_hidden")


def _lemma_dict(L) -> dict:
    return {
        "M": L.M,
        "r": L.r,
        "k": L.k,
        "A": L.A,
        "B": L.B,
        "n_train": L.n_train,
        "kind": L.kind,
        "a": L.a,
        "b": L.b,
    }


def _top(c) -> dict:
    return {
        "M": c.M,
        "cover": c.train_cover_frac,
        "n_lemmas": len(c.lemmas),
        "uncovered": list(c.uncovered),
        "k_used": sorted({L.k for L in c.lemmas}),
        "kinds": sorted({L.kind for L in c.lemmas}),
    }


def _select(bb, ranked: list, n_hi: int) -> tuple[object, dict, list]:
    """Falsifier-in-the-loop: smallest complete covering that survives exact testing."""
    attempts = []
    complete = [c for c in ranked if not c.uncovered]
    complete.sort(key=lambda c: (c.M, max((L.k for L in c.lemmas), default=99), len(c.lemmas)))
    for cert in complete:
        ver = verify(bb, cert, n_hi=n_hi)
        attempts.append(
            {
                "M": cert.M,
                "max_k": max((L.k for L in cert.lemmas), default=None),
                "passed": ver["passed_falsifier"],
                "level": ver["level"],
            }
        )
        if ver["passed_falsifier"]:
            return cert, ver, attempts
    best = ranked[0]
    return best, verify(bb, best, n_hi=n_hi), attempts


def discover_one(map_name: str, run: Path) -> dict:
    print(f"discover {map_name}", flush=True)
    bb = get_blackbox(map_name)
    ranked = sweep_moduli(bb, n_max=N_TRAIN)
    best, ver, attempts = _select(bb, ranked, N_FALSIFY)
    uni = (ver.get("universal") or {}).get("proven")
    print(
        f"  M={best.M} cover={best.train_cover_frac:.4f} "
        f"uncovered={list(best.uncovered)} {ver['level']} "
        f"falsifier={ver['passed_falsifier']} universal={uni}",
        flush=True,
    )
    neural = {"skipped": True, "reason": "BRIDGE_NEURAL not set; login node has no GPU"}
    want_neural = os.environ.get("BRIDGE_NEURAL", "").strip() in {"1", "true", "yes"}
    if want_neural:
        try:
            from src.bridge.neural_potential import train_population

            ns = list(range(3, N_TRAIN + 1, 2))
            fs = [bb.F_odd(n) for n in ns]
            neural = train_population(ns, fs, pop=64, steps=400, seed=SEED)
        except ImportError:
            neural = {"skipped": True, "reason": "torch not available"}
    payload = {
        "status": "exploratory",
        "map": map_name,
        "n_train": N_TRAIN,
        "n_falsify": N_FALSIFY,
        "seed": SEED,
        "best_M": best.M,
        "best_cover": best.train_cover_frac,
        "uncovered": list(best.uncovered),
        "lemmas": [_lemma_dict(L) for L in best.lemmas],
        "verifier": ver,
        "falsifier_attempts": attempts,
        "neural_potential": neural,
        "top_moduli": [_top(c) for c in ranked[:8]],
        "minimality": _top(best) if (not best.uncovered and ver["passed_falsifier"]) else _minimality(ranked),
    }
    dest = run / map_name
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "summary.json").write_text(json.dumps(payload, indent=2))
    return payload


def _minimality(ranked) -> dict | None:
    complete = [c for c in ranked if not c.uncovered and c.train_cover_frac >= 1.0 - 1e-12]
    if not complete:
        return None
    best = min(complete, key=lambda c: (c.M, len(c.lemmas), max((L.k for L in c.lemmas), default=99)))
    return _top(best)


def _compare(discovered: dict, oracle_path: Path) -> dict:
    if not oracle_path.is_file():
        return {"status": "missing_oracle", "path": str(oracle_path)}
    oracle = json.loads(oracle_path.read_text())
    cert = oracle.get("certificate") or {}
    o_lemmas = cert.get("lemmas") or []
    d_lemmas = discovered["lemmas"]
    o_M = cert.get("modulus")
    d_M = discovered["best_M"]
    o_k = sorted({L["k"] for L in o_lemmas})
    d_k = sorted({L["k"] for L in d_lemmas})
    d_kinds = sorted({L["kind"] for L in d_lemmas})
    passed = bool(discovered["verifier"]["passed_falsifier"] and not discovered["uncovered"])
    o_ok = bool(cert.get("verified"))
    if passed and o_ok:
        if d_M < o_M:
            relation = "smaller_covering"
        elif d_M == o_M and d_k == o_k and d_kinds == ["affine"]:
            relation = "same_coordinates"
        else:
            relation = "equivalent_different_coordinates"
    elif (not passed) and (not o_ok):
        relation = "agreed_negative"
    elif passed and not o_ok:
        relation = "discoverer_covered_oracle_did_not"
    else:
        relation = "discoverer_failed_oracle_verified"
    return {
        "status": "compared",
        "oracle_verified": o_ok,
        "oracle_M": o_M,
        "oracle_horizons": o_k,
        "oracle_n_lemmas": len(o_lemmas),
        "oracle_n_failures": len(cert.get("failures") or []),
        "discovered_passed": passed,
        "discovered_M": d_M,
        "discovered_horizons": d_k,
        "discovered_kinds": d_kinds,
        "discovered_n_lemmas": len(d_lemmas),
        "relation": relation,
        "level": discovered["verifier"]["level"],
    }


def main() -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run = Path("tiny_tao_results") / f"bridge_suite_{stamp}"
    run.mkdir(parents=True, exist_ok=True)
    discovered = []
    for name in MAPS:
        discovered.append(discover_one(name, run))
    comparisons = []
    for payload in discovered:
        comparisons.append(
            {
                "map": payload["map"],
                **_compare(payload, ORACLE_DIR / f"{payload['map']}.json"),
            }
        )
    index = {
        "status": "exploratory",
        "run": str(run),
        "n_train": N_TRAIN,
        "n_falsify": N_FALSIFY,
        "seed": SEED,
        "neural": "skipped",
        "maps": [
            {
                "map": p["map"],
                "best_M": p["best_M"],
                "cover": p["best_cover"],
                "uncovered": p["uncovered"],
                "level": p["verifier"]["level"],
                "passed": p["verifier"]["passed_falsifier"],
                "universal": (p["verifier"].get("universal") or {}).get("proven"),
                "minimality": p["minimality"],
            }
            for p in discovered
        ],
        "oracle_comparison": comparisons,
    }
    (run / "index.json").write_text(json.dumps(index, indent=2))
    print(json.dumps(index, indent=2))
    print("wrote", run)


if __name__ == "__main__":
    main()
