"""Model T on gcd: preregistered valuation-min certificate.

Fixed coordinatewise min, locked primes {2,3,5,7,11}, N=12.
Does not add primes, valuation-init, Euclid/Stein decoders, or loosen
exactness if n_certified=0. Does not train Collatz or Model C.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.analysis.stats import wilson_interval
from src.analysis.valuation_min import HEADS, N, PRIMES, certify_net, decoder_sanity
from src.config import ExperimentConfig
from src.hardware import resolve_device
from src.repro import environment_record, seed_everything
from src.runner import finalize_population, fit_and_build, prepare_from_task, train_prepared
from src.storage import new_run_dir, write_json
from src.tasks.gcd import DESTROY_SEED, gcd_destroyed, gcd_meet, held_row_coverage


PREREG = ROOT / "tiny_tao_results" / "preregistration_gcd_valuation_min.json"
POPULATION = 1024
EPOCHS = 8000
LR = 3e-3
SEED = 0
A0 = 6


def make_cfg(task_name: str) -> ExperimentConfig:
    cfg = ExperimentConfig.from_yaml(ROOT / "configs" / "held_row.yaml")
    cfg.task.name = task_name
    cfg.task.p = N
    cfg.task.split = "held_row"
    cfg.task.held_operands = [A0]
    cfg.model.interaction = "tropical"
    cfg.model.hidden_dims = [HEADS]
    cfg.model.activation = "identity"
    cfg.population.size = POPULATION
    cfg.train.epochs = EPOCHS
    cfg.train.lr = LR
    cfg.seed = SEED
    cfg.run_name = f"tropical_h{HEADS}_{task_name}_held_row"
    return cfg


def score_exact(model, exact_idx: torch.Tensor, table: np.ndarray) -> dict:
    if exact_idx.numel() == 0:
        return {
            "n_exact": 0,
            "n_certified": 0,
            "n_sum_also_exact": 0,
            "breaks": {},
            "certified_nets": [],
        }
    E = model.E.detach().cpu()[exact_idx].numpy().astype(np.float64)
    certs = []
    for i in range(E.shape[0]):
        cert = certify_net(E[i], table, scramble_seed=int(exact_idx[i]))
        cert["net"] = int(exact_idx[i])
        certs.append(cert)
    n_certified = int(sum(1 for c in certs if c["certified"]))
    n_sum = int(sum(1 for c in certs if c.get("sum_n_correct") == N * N))
    breaks: dict[str, int] = {}
    for c in certs:
        if not c["certified"]:
            b = str(c.get("break_at"))
            breaks[b] = breaks.get(b, 0) + 1
    return {
        "n_exact": int(exact_idx.numel()),
        "n_certified": n_certified,
        "n_sum_also_exact": n_sum,
        "breaks": breaks,
        "certified_nets": [c for c in certs if c["certified"]][:16],
    }


def run_table(task, cfg: ExperimentConfig, device: torch.device, parent: Path) -> dict:
    coverage = held_row_coverage(task, A0)
    if not coverage["all_classes_in_train"] or not coverage["a0_appears_as_b_in_train"]:
        raise RuntimeError(coverage)
    seed_everything(SEED)
    prepared = prepare_from_task(task, cfg, device, protocol="split")
    model, fitted = fit_and_build(cfg, prepared, device, seed=SEED)
    sub = parent / cfg.run_name
    for d in ("metrics", "checkpoints", "figures", "logs"):
        (sub / d).mkdir(parents=True, exist_ok=True)
    print(
        f"{task.name} n={N} H={HEADS} primes={list(PRIMES)} P={fitted} "
        f"params={model.n_params_per_network()} device={device}",
        flush=True,
    )
    result = train_prepared(model, prepared, cfg, show_progress=True)
    packed = finalize_population(
        model, result, prepared, sub, write_figures=True, max_exact_saved=fitted
    )
    exact_mask = packed["labels"] == 4
    scored = score_exact(model, torch.where(exact_mask.detach().cpu())[0], task.table)
    frac, lo, hi = wilson_interval(scored["n_exact"], fitted)
    payload = {
        "task": task.name,
        "n": N,
        "heads": HEADS,
        "primes": list(PRIMES),
        "params": model.n_params_per_network(),
        "population": fitted,
        "epochs": EPOCHS,
        "seconds": result.seconds,
        "coverage": coverage,
        "frac_exact": frac,
        "frac_exact_lo": lo,
        "frac_exact_hi": hi,
        **scored,
    }
    write_json(sub / "metrics" / "valuation_min.json", payload)
    print(json.dumps({k: payload[k] for k in (
        "task", "n_exact", "n_certified", "n_sum_also_exact", "breaks", "seconds",
    )}, indent=2), flush=True)
    return payload


def main() -> None:
    pre = json.loads(PREREG.read_text())
    assert pre["status"] == "preregistered"
    sanity = decoder_sanity()
    print(
        "synthetic decoder meet", sanity["meet"]["certified"],
        "destroyed", sanity["destroyed"]["certified"],
        "passed", sanity["passed"],
        flush=True,
    )
    if not sanity["passed"]:
        raise RuntimeError(f"decoder sanity failed before training: {sanity}")

    device = resolve_device("cuda")
    parent = new_run_dir(ROOT, "gcd_valuation_min")
    write_json(parent / "predictions.json", pre)
    write_json(parent / "decoder_sanity.json", {
        "passed": sanity["passed"],
        "meet_certified": sanity["meet"]["certified"],
        "destroyed_certified": sanity["destroyed"]["certified"],
        "destroyed_equals_gcd": sanity["destroyed_equals_gcd"],
        "destroy_seed": DESTROY_SEED,
    })
    write_json(parent / "environment.json", environment_record())

    meet = run_table(gcd_meet(N), make_cfg("gcd_meet"), device, parent)
    destroyed = run_table(gcd_destroyed(N, DESTROY_SEED), make_cfg("gcd_destroyed"), device, parent)

    p3_vacuous = meet["n_certified"] == 0
    p3 = meet["n_sum_also_exact"] == 0
    score = {
        "P1_exact_exists_meet": {"verdict": "hold" if meet["n_exact"] >= 1 else "miss", "n_exact": meet["n_exact"]},
        "P2_certified_not_guaranteed": {
            "verdict": "hold",
            "n_certified": meet["n_certified"],
            "note": "Appeared" if meet["n_certified"] else "Did not appear; valid miss",
        },
        "P3_sum_never_certifies_min_nets": {
            "verdict": "vacuous" if p3_vacuous else ("hold" if p3 else "miss"),
            "n_sum_also_exact": meet["n_sum_also_exact"],
        },
        "P4_destroyed_zero_certified": {
            "verdict": "hold" if destroyed["n_certified"] == 0 else "miss",
            "n_certified": destroyed["n_certified"],
            "n_exact": destroyed["n_exact"],
        },
        "P5_do_not_add_primes": {
            "verdict": "not_triggered" if meet["n_certified"] else "applied",
            "note": "n_certified=0: do not add primes, raise H, valuation-init, or Euclid/Stein",
        },
        "P6_not_collatz": {"verdict": "applied"},
        "P7_not_model_c": {"verdict": "applied"},
    }
    scientific = meet["n_certified"] > 0 and destroyed["n_certified"] == 0
    scored = {
        "status": "scored",
        "preregistration": str(PREREG),
        "run": str(parent),
        "scored_after_training": True,
        "protocol": f"Model T H={HEADS}, N={N}, held_row a0={A0}, P={POPULATION}, {EPOCHS} Adam steps, lr={LR}, seed={SEED}, no valuation init",
        "params": 132,
        "meet": meet,
        "destroyed": destroyed,
        "predictions": score,
        "scientific_valuation_min": scientific,
        "lesson": (
            "Model T is min of real heads, not complex multiply. "
            "Certified means neural-free valuation-meet, drop-one-prime fails, sum fails. "
            "Destroyed-law control must not certify. Do not add primes."
        ),
    }
    write_json(parent / "metrics" / "valuation_min.json", scored)
    out_score = ROOT / "tiny_tao_results" / "preregistration_gcd_valuation_min.scored.json"
    write_json(out_score, scored)
    print("SCIENTIFIC VALUATION-MIN:", scientific)
    print("wrote", parent)
    print("scored", out_score)


if __name__ == "__main__":
    main()
