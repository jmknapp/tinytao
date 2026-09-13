"""Discoverer–Falsifier on the approved medium bridge map. Does not read oracle JSON."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from src.bridge.blackbox import get_blackbox
from src.bridge.discoverer import sweep_moduli
from src.bridge.verifier import verify


MAP_NAME = "medium_five_on_1_mod_16"
N_TRAIN = 400
N_FALSIFY = 20_000
SEED = 20260909


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


def main() -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run = Path("tiny_tao_results") / f"bridge_discover_{MAP_NAME}_{stamp}"
    run.mkdir(parents=True, exist_ok=True)

    print(f"discover {MAP_NAME} n_train={N_TRAIN} n_falsify={N_FALSIFY}", flush=True)
    bb = get_blackbox(MAP_NAME)
    ranked = sweep_moduli(bb, n_max=N_TRAIN)
    best = ranked[0]
    print(
        f"best M={best.M} cover={best.train_cover_frac:.4f} "
        f"lemmas={len(best.lemmas)} uncovered={list(best.uncovered)} "
        f"kinds={sorted({L.kind for L in best.lemmas})}",
        flush=True,
    )
    ver = verify(bb, best, n_hi=N_FALSIFY)
    print(f"verifier {ver['level']} passed={ver['passed_falsifier']}", flush=True)

    neural = {"skipped": True, "reason": "BRIDGE_NEURAL not set"}
    want_neural = os.environ.get("BRIDGE_NEURAL", "").strip() in {"1", "true", "yes"}
    if want_neural:
        try:
            from src.bridge.neural_potential import train_population
            ns, fs = [], []
            for n in range(3, N_TRAIN + 1, 2):
                ns.append(n)
                fs.append(bb.F_odd(n))
            print(f"neural pop=64 steps=400 n={len(ns)}", flush=True)
            neural = train_population(ns, fs, pop=64, steps=400, seed=SEED)
        except ImportError:
            neural = {"skipped": True, "reason": "torch not available"}

    payload = {
        "status": "exploratory",
        "map": MAP_NAME,
        "n_train": N_TRAIN,
        "n_falsify": N_FALSIFY,
        "seed": SEED,
        "best_M": best.M,
        "best_cover": best.train_cover_frac,
        "uncovered": list(best.uncovered),
        "lemmas": [_lemma_dict(L) for L in best.lemmas],
        "verifier": ver,
        "neural_potential": neural,
        "top_moduli": [
            {
                "M": c.M,
                "cover": c.train_cover_frac,
                "n_lemmas": len(c.lemmas),
                "uncovered": list(c.uncovered),
                "k_used": sorted({L.k for L in c.lemmas}),
                "kinds": sorted({L.kind for L in c.lemmas}),
            }
            for c in ranked[:8]
        ],
    }
    (run / "summary.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps({k: payload[k] for k in ("map", "best_M", "best_cover", "verifier")}, indent=2))
    print("neural:", "skipped" if neural.get("skipped") else "ok")
    if not neural.get("skipped"):
        en = neural["modulus_weight_energy"]
        for m, e in sorted(en.items(), key=lambda kv: -kv[1])[:8]:
            print(f"  mod {m:>3}: {e:.4f}")
    print("wrote", run)


if __name__ == "__main__":
    main()
