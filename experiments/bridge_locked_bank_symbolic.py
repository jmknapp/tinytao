"""15.6: exact extractor on the locked feature bank. No neural stage. Oracle last."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.bridge.blackbox import get_blackbox
from src.bridge.discoverer import sweep_moduli
from src.bridge.neural_lemmas import K_MAX
from src.bridge.neural_potential import FEATURE_MODULI
from src.bridge.verifier import verify


MAPS = (
    "prospective_seven_and_three_mod_32",
    "medium_five_on_1_mod_16",
    "hard_five_on_1_mod_16",
    "control_five_on_1_mod_4",
    "prospective_seven_on_1_mod_32",
)
N_TRAIN = 2000
N_FALSIFY = 20_000
SEED = 20260909
PREREG = Path("tiny_tao_results/bridge_preregistration_locked_bank_symbolic.json")
ORACLE_DIR = Path("tiny_tao_results/bridge_oracle_hidden")


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


def _lemma_at(payload: dict, r: int) -> dict | None:
    return next((L for L in payload["lemmas"] if L["r"] == r), None)


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
    m16 = next((c for c in ranked if c.M == 16), None)
    cmp_ = _compare(chosen, ver, ORACLE_DIR / f"{map_name}.json")
    print("oracle", cmp_, flush=True)
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
        "M16_uncovered": list(m16.uncovered) if m16 is not None else None,
        "chosen_M": chosen.M,
        "lemmas": [_lemma_dict(L) for L in chosen.lemmas],
        "uncovered": list(chosen.uncovered),
        "verifier": ver,
        "falsifier_attempts": attempts,
        "level5_gate": level5,
        "oracle": cmp_,
    }
    dest = run / map_name
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "summary.json").write_text(json.dumps(payload, indent=2))
    return payload


def main() -> None:
    pre = json.loads(PREREG.read_text())
    assert pre["status"] == "preregistered"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run = Path("tiny_tao_results") / f"bridge_locked_bank_symbolic_{stamp}"
    run.mkdir(parents=True, exist_ok=True)
    results = [discover_one(name, run) for name in MAPS]
    thicker = next(p for p in results if p["map"] == "prospective_seven_and_three_mod_32")
    medium = next(p for p in results if p["map"] == "medium_five_on_1_mod_16")
    hard = next(p for p in results if p["map"] == "hard_five_on_1_mod_16")
    control = next(p for p in results if p["map"] == "control_five_on_1_mod_4")
    thin = next(p for p in results if p["map"] == "prospective_seven_on_1_mod_32")
    t_r1 = _lemma_at(thicker, 1)
    m_r1 = _lemma_at(medium, 1)
    h_r9 = _lemma_at(hard, 9)
    n_r1 = _lemma_at(thin, 1)
    p1 = (
        thicker["level5_gate"]
        and thicker["chosen_M"] == 32
        and t_r1 is not None
        and t_r1["kind"] == "affine"
        and t_r1["k"] == 1
        and t_r1["A"] == 28
        and t_r1["B"] == 1
    )
    p2 = (
        medium["level5_gate"]
        and medium["chosen_M"] == 16
        and m_r1 is not None
        and m_r1["kind"] == "affine"
        and m_r1["k"] == 2
        and m_r1["A"] == 10
        and m_r1["B"] == 1
    )
    p3 = (
        hard["level5_gate"]
        and hard["chosen_M"] == 16
        and h_r9 is not None
        and h_r9["kind"] == "affine"
        and h_r9["A"] == 12
        and h_r9["B"] == 7
    )
    p4 = not control["level5_gate"]
    p5 = (
        thin["level5_gate"]
        and thin["chosen_M"] == 32
        and 1 in (thin["M16_uncovered"] or [])
        and n_r1 is not None
        and n_r1["kind"] == "affine"
        and n_r1["k"] == 1
        and n_r1["A"] == 28
        and n_r1["B"] == 1
    )
    score = {
        "P1_thicker_level5": p1,
        "P2_medium_level5": p2,
        "P3_hard_level5": p3,
        "P4_control_fails": p4,
        "P5_thin_level5": p5,
        "P6_heads_not_required": p1 and p2 and p3 and p4 and p5,
        "preregistered_gate": p1 and p2 and p3 and p4 and p5,
        "thicker_r1": t_r1,
        "medium_r1": m_r1,
        "hard_r9": h_r9,
        "thin_r1": n_r1,
        "thin_M16_uncovered": thin["M16_uncovered"],
    }
    index = {
        "status": "exploratory",
        "run": str(run),
        "preregistration": str(PREREG),
        "neural": False,
        "score": score,
        "maps": [
            {
                "map": p["map"],
                "chosen_M": p["chosen_M"],
                "level": p["verifier"]["level"],
                "level5_gate": p["level5_gate"],
                "uncovered": p["uncovered"],
                "M16_uncovered": p["M16_uncovered"],
                "oracle": p["oracle"],
            }
            for p in results
        ],
    }
    (run / "index.json").write_text(json.dumps(index, indent=2))
    print(json.dumps(index, indent=2))
    print("PREREGISTERED GATE (P1-P5):", score["preregistered_gate"])


if __name__ == "__main__":
    main()
