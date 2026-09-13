"""Neural affine-head Discoverer on the locked feature bank. Oracle JSON last."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.bridge.blackbox import get_blackbox
from src.bridge.neural_lemmas import propose_neural_certificates
from src.bridge.neural_potential import FEATURE_MODULI
from src.bridge.verifier import verify


MAPS = (
    "prospective_seven_and_three_mod_32",
    "medium_five_on_1_mod_16",
    "control_five_on_1_mod_4",
)
N_TRAIN = 2000
N_FALSIFY = 20_000
POP = 256
STEPS = 800
LR = 5e-2
SEED = 20260909
PREREG = Path("tiny_tao_results/bridge_preregistration_neural_lemmas.json")
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
    ns = list(range(3, N_TRAIN + 1, 2))
    certs, fits = propose_neural_certificates(
        bb, ns,
        moduli=FEATURE_MODULI,
        pop=POP, steps=STEPS, lr=LR, seed=SEED,
        require_cuda=True, log_every=200,
    )
    chosen, ver, attempts = _select(bb, certs, N_FALSIFY)
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
    cmp_ = _compare(chosen, ver, ORACLE_DIR / f"{map_name}.json")
    print("oracle", cmp_, flush=True)
    payload = {
        "status": "exploratory",
        "preregistration": str(PREREG),
        "map": map_name,
        "n_train": N_TRAIN,
        "n_falsify": N_FALSIFY,
        "seed": SEED,
        "pop": POP,
        "steps": STEPS,
        "feature_moduli": list(FEATURE_MODULI),
        "fits": fits,
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
    run = Path("tiny_tao_results") / f"bridge_neural_lemmas_{stamp}"
    run.mkdir(parents=True, exist_ok=True)
    results = [discover_one(name, run) for name in MAPS]
    thicker = next(p for p in results if p["map"] == "prospective_seven_and_three_mod_32")
    medium = next(p for p in results if p["map"] == "medium_five_on_1_mod_16")
    control = next(p for p in results if p["map"] == "control_five_on_1_mod_4")
    score = {
        "P1_thicker_level5": thicker["level5_gate"] and thicker["chosen_M"] == 32,
        "P1_thicker_level5_any_M": thicker["level5_gate"],
        "P2_medium_still_level5": medium["level5_gate"],
        "P3_control_still_fails": not control["level5_gate"],
        "preregistered_gate": thicker["level5_gate"],
    }
    index = {
        "status": "exploratory",
        "run": str(run),
        "preregistration": str(PREREG),
        "score": score,
        "maps": [
            {
                "map": p["map"],
                "chosen_M": p["chosen_M"],
                "level": p["verifier"]["level"],
                "level5_gate": p["level5_gate"],
                "uncovered": p["uncovered"],
                "oracle": p["oracle"],
            }
            for p in results
        ],
    }
    (run / "index.json").write_text(json.dumps(index, indent=2))
    print(json.dumps(index, indent=2))
    print("PREREGISTERED GATE (P1 any Level 5 on thicker):", score["preregistered_gate"])


if __name__ == "__main__":
    main()
