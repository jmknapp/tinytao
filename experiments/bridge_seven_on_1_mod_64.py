"""15.7: locked-bank exact sweep on a covering that lives at M=64. Oracle last."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.bridge.blackbox import get_blackbox
from src.bridge.catalog import get_bridge_map
from src.bridge.discoverer import sweep_moduli
from src.bridge.neural_lemmas import K_MAX
from src.bridge.neural_potential import FEATURE_MODULI
from src.bridge.oracle import certificate_to_json, prove_eventual_descent
from src.bridge.verifier import verify


MAPS = (
    "prospective_seven_on_1_mod_64",
    "medium_five_on_1_mod_16",
    "control_five_on_1_mod_4",
)
N_TRAIN = 2000
N_FALSIFY = 20_000
SEED = 20260909
PREREG = Path("tiny_tao_results/bridge_preregistration_seven_on_1_mod_64.json")
ORACLE_DIR = Path("tiny_tao_results/bridge_oracle_hidden")
NEW_MAP = "prospective_seven_on_1_mod_64"


def _lemma_dict(L) -> dict:
    return {
        "M": L.M,
        "r": L.r,
        "k": L.k,
        "A": L.A,
        "B": L.B,
        "kind": L.kind,
        "a": L.a,
        "b": L.b,
        "n_train": L.n_train,
        "source": L.source,
    }


def _select(bb, ranked, n_hi):
    attempts = []
    complete = [c for c in ranked if not c.uncovered]
    complete.sort(key=lambda c: (c.M, max((L.k for L in c.lemmas), default=99), len(c.lemmas)))
    for cert in complete:
        ver = verify(bb, cert, n_hi=n_hi)
        attempts.append({"M": cert.M, "passed": ver["passed_falsifier"], "level": ver["level"]})
        print(
            f"  try M={cert.M}  cover={cert.train_cover_frac:.3f}  "
            f"{ver['level']}  falsifier={ver['passed_falsifier']}",
            flush=True,
        )
        if ver["passed_falsifier"]:
            return cert, ver, attempts
    best = ranked[0]
    return best, verify(bb, best, n_hi=n_hi), attempts


def _compare(chosen, ver, oracle_path: Path) -> dict:
    if not oracle_path.is_file():
        return {"status": "missing_oracle"}
    oracle = json.loads(oracle_path.read_text())
    oc = oracle.get("certificate") or {}
    o_ok = bool(oc.get("verified"))
    passed = bool(ver["passed_falsifier"]) and not chosen.uncovered
    if passed and o_ok:
        relation = "smaller_covering" if chosen.M < oc["modulus"] else "covering_found"
    elif (not passed) and (not o_ok):
        relation = "agreed_negative"
    elif passed and not o_ok:
        relation = "discoverer_covered_oracle_did_not"
    else:
        relation = "discoverer_failed_oracle_verified"
    return {
        "oracle_verified": o_ok,
        "oracle_M": oc.get("modulus"),
        "relation": relation,
    }


def discover_one(map_name: str, run: Path) -> dict:
    print(f"\n=== {map_name} ===", flush=True)
    bb = get_blackbox(map_name)
    ranked = sweep_moduli(bb, n_max=N_TRAIN, moduli=FEATURE_MODULI, k_max=K_MAX)
    chosen, ver, attempts = _select(bb, ranked, N_FALSIFY)
    level5 = ver["level"] == "LEVEL_5_universal_descent" and ver["passed_falsifier"]
    print(
        f"chosen M={chosen.M}  {ver['level']}  uncovered={list(chosen.uncovered)}  "
        f"LEVEL_5={level5}",
        flush=True,
    )
    for L in chosen.lemmas:
        if L.kind == "odd_part":
            print(f"  r={L.r:>3}  odd_part({L.a}n+{L.b})")
        else:
            print(f"  r={L.r:>3}  k={L.k}  F^k={L.A}q+{L.B}")
    m32 = next((c for c in ranked if c.M == 32), None)
    payload = {
        "status": "exploratory",
        "preregistration": str(PREREG),
        "map": map_name,
        "neural": False,
        "n_train": N_TRAIN,
        "n_falsify": N_FALSIFY,
        "seed": SEED,
        "k_max": K_MAX,
        "feature_moduli": list(FEATURE_MODULI),
        "moduli_cover": [
            {"M": c.M, "cover": c.train_cover_frac, "uncovered": list(c.uncovered)}
            for c in ranked
        ],
        "M32_uncovered": list(m32.uncovered) if m32 is not None else None,
        "chosen_M": chosen.M,
        "lemmas": [_lemma_dict(L) for L in chosen.lemmas],
        "uncovered": list(chosen.uncovered),
        "verifier": ver,
        "falsifier_attempts": attempts,
        "level5_gate": level5,
        "oracle": {"status": "deferred"},
    }
    dest = run / map_name
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "summary.json").write_text(json.dumps(payload, indent=2))
    return payload


def _write_oracle_after_discovery(map_name: str) -> dict:
    m = get_bridge_map(map_name)
    cert = prove_eventual_descent(m, max_horizon=12, small_upto=512)
    payload = {
        "name": m.name,
        "halt": m.halt,
        "rule_modulus": m.modulus,
        "rules": [{"r": r.r, "m": r.m, "a": r.a, "b": r.b} for r in m.rules],
        "certificate": certificate_to_json(cert),
        "written_after_discovery": True,
    }
    path = ORACLE_DIR / f"{map_name}.json"
    path.write_text(json.dumps(payload, indent=2))
    index_path = ORACLE_DIR / "index.json"
    index = json.loads(index_path.read_text()) if index_path.is_file() else []
    index = [row for row in index if row.get("name") != map_name]
    index.append(
        {
            "name": m.name,
            "verified": cert.verified,
            "modulus": cert.modulus,
            "n_lemmas": len(cert.lemmas),
            "n_failures": len(cert.failures),
            "horizons": sorted({L.k for L in cert.lemmas}),
            "path": str(path),
            "written_after_discovery": True,
        }
    )
    index_path.write_text(json.dumps(index, indent=2))
    print(
        f"oracle after discovery: {map_name} verified={cert.verified} M={cert.modulus}",
        flush=True,
    )
    return payload


def main() -> None:
    pre = json.loads(PREREG.read_text())
    assert pre["status"] == "preregistered"
    assert 64 not in FEATURE_MODULI
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run = Path("tiny_tao_results") / f"bridge_seven_on_1_mod_64_{stamp}"
    run.mkdir(parents=True, exist_ok=True)
    results = [discover_one(name, run) for name in MAPS]
    _write_oracle_after_discovery(NEW_MAP)
    for payload in results:
        dest = run / payload["map"]
        chosen_like = type("C", (), {"M": payload["chosen_M"], "uncovered": payload["uncovered"]})()
        payload["oracle"] = _compare(chosen_like, payload["verifier"], ORACLE_DIR / f"{payload['map']}.json")
        (dest / "summary.json").write_text(json.dumps(payload, indent=2))
        print(payload["map"], "oracle", payload["oracle"], flush=True)
    out = next(p for p in results if p["map"] == NEW_MAP)
    medium = next(p for p in results if p["map"] == "medium_five_on_1_mod_16")
    control = next(p for p in results if p["map"] == "control_five_on_1_mod_4")
    residue1_uncovered_all = all(
        1 in row["uncovered"] for row in out["moduli_cover"] if row["uncovered"] is not None
    )
    p1 = (not out["level5_gate"]) and (1 in (out["M32_uncovered"] or [])) and residue1_uncovered_all
    p2 = medium["level5_gate"] and medium["chosen_M"] == 16
    p3 = (not control["level5_gate"]) and control["oracle"].get("relation") == "agreed_negative"
    p4 = (
        out["oracle"].get("oracle_verified") is True
        and out["oracle"].get("oracle_M") == 64
        and out["oracle"].get("relation") == "discoverer_failed_oracle_verified"
    )
    score = {
        "P1_out_of_bank_fails": p1,
        "P2_medium_still_level5": p2,
        "P3_control_still_fails": p3,
        "P4_oracle_verifies_at_64": p4,
        "P5_bank_still_locked": 64 not in FEATURE_MODULI,
        "preregistered_gate": p1 and p2 and p3 and p4,
        "out_chosen_M": out["chosen_M"],
        "out_uncovered": out["uncovered"],
        "M32_uncovered": out["M32_uncovered"],
    }
    index = {
        "status": "exploratory",
        "run": str(run),
        "preregistration": str(PREREG),
        "neural": False,
        "feature_moduli": list(FEATURE_MODULI),
        "score": score,
        "maps": [
            {
                "map": p["map"],
                "chosen_M": p["chosen_M"],
                "level": p["verifier"]["level"],
                "level5_gate": p["level5_gate"],
                "uncovered": p["uncovered"],
                "M32_uncovered": p["M32_uncovered"],
                "oracle": p["oracle"],
            }
            for p in results
        ],
    }
    (run / "index.json").write_text(json.dumps(index, indent=2))
    print(json.dumps(index, indent=2))
    print("PREREGISTERED GATE (P1-P4):", score["preregistered_gate"])


if __name__ == "__main__":
    main()
