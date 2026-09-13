"""H=3 Model C at p=31: preregistered 3-torus CRT certificate.

Writes the preregistration (already on disk) check, trains, then scores.
Does not raise H, dlog-init, or loosen exactness if n_certified=0.
"""

from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from experiments.stage_a_primes import held_row_coverage
from src.analysis.fourier import discrete_log_table, primitive_root
from src.analysis.group_law import fit_dlog_circle_batch
from src.analysis.level4_h1 import effective_winding
from src.analysis.stats import wilson_interval
from src.analysis.three_torus import (
    HEADS,
    N,
    P,
    decoder_sanity,
    image_order,
    certify_net,
)
from src.config import ExperimentConfig
from src.hardware import resolve_device
from src.repro import environment_record, seed_everything
from src.runner import finalize_population, fit_and_build, prepare_task, train_prepared
from src.storage import new_run_dir, write_json


PREREG = ROOT / "tiny_tao_results" / "preregistration_p31_h3_three_torus.json"
POPULATION = 1024
EPOCHS = 8000
LR = 3e-3
SEED = 0


def make_cfg() -> ExperimentConfig:
    cfg = ExperimentConfig.from_yaml(ROOT / "configs" / "held_row.yaml")
    cfg.task.name = "modular_multiplication_star"
    cfg.task.p = P
    cfg.task.split = "held_row"
    cfg.task.held_operands = [-1]
    cfg.model.interaction = "complex"
    cfg.model.hidden_dims = [HEADS]
    cfg.model.activation = "identity"
    cfg.population.size = POPULATION
    cfg.train.epochs = EPOCHS
    cfg.train.lr = LR
    cfg.seed = SEED
    cfg.run_name = "complex_h3_p31_held_row"
    return cfg


def main() -> None:
    pre = json.loads(PREREG.read_text())
    assert pre["status"] == "preregistered"
    sanity = decoder_sanity()
    print("synthetic CRT decoder", sanity["passed"], sanity["cert"].get("break_at"), flush=True)
    if not sanity["passed"]:
        raise RuntimeError(f"decoder sanity failed before training: {sanity}")

    device = resolve_device("cuda")
    cfg = make_cfg()
    a0 = P - 1
    coverage = held_row_coverage(P, a0)
    print("coverage", coverage, flush=True)
    if not coverage["all_classes_in_train"] or not coverage["a0_appears_as_b_in_train"]:
        raise RuntimeError(coverage)

    parent = new_run_dir(ROOT / "results" / "runs", "p31_h3_three_torus")
    write_json(parent / "predictions.json", pre)
    write_json(parent / "decoder_sanity.json", {"passed": sanity["passed"], "cert": sanity["cert"]})
    write_json(parent / "environment.json", environment_record())

    seed_everything(SEED)
    prepared = prepare_task(cfg, device, protocol="split")
    model, fitted = fit_and_build(cfg, prepared, device, seed=SEED)
    sub = parent / cfg.run_name
    for d in ("metrics", "checkpoints", "figures", "logs"):
        (sub / d).mkdir(parents=True, exist_ok=True)
    print(
        f"p={P} H={HEADS} P={fitted} params={model.n_params_per_network()} "
        f"g={primitive_root(P)} device={device}",
        flush=True,
    )
    result = train_prepared(model, prepared, cfg, show_progress=True)
    packed = finalize_population(
        model, result, prepared, sub, write_figures=True, max_exact_saved=fitted
    )
    exact_mask = packed["labels"] == 4
    n_exact = int(exact_mask.sum())
    idx = torch.where(exact_mask.detach().cpu())[0]
    dlog = discrete_log_table(P)
    g = primitive_root(P)
    certs = []
    n_faithful_head = 0
    gcd_hist: dict[str, int] = {}
    if n_exact:
        E = model.E.detach().cpu()[idx].numpy().astype(np.float64)
        k_effs = []
        phis = []
        for h in range(HEADS):
            batch = fit_dlog_circle_batch(E[:, :, h, :], dlog, N)
            k_effs.append(effective_winding(batch["k"], batch["conjugate"], N))
            phis.append(batch["phi"])
        k_effs = np.stack(k_effs, axis=1)
        phis = np.stack(phis, axis=1)
        for i in range(n_exact):
            cert = certify_net(E[i], k_effs[i], phis[i], p=P, g=g)
            cert["net"] = int(idx[i])
            certs.append(cert)
            if cert["has_faithful_head"]:
                n_faithful_head += 1
            key = ",".join(str(x) for x in sorted(cert["gcds"]))
            gcd_hist[key] = gcd_hist.get(key, 0) + 1

    n_certified = int(sum(1 for c in certs if c["certified"]))
    n_candidate = int(sum(1 for c in certs if c["candidate_triple"]))
    frac, lo, hi = wilson_interval(n_exact, fitted)
    score = {
        "P1_exact_exists": n_exact >= 1,
        "P2_f_like_present": n_faithful_head >= 1,
        "P3_torus_not_guaranteed": True,
        "P4_h1_cannot": True,
        "P5_do_not_add_heads": n_certified == 0,
        "n_exact": n_exact,
        "n_population": fitted,
        "n_candidate_triple": n_candidate,
        "n_certified": n_certified,
        "n_with_faithful_head": n_faithful_head,
        "scientific_3_torus": n_certified > 0,
    }
    payload = {
        "status": "exploratory",
        "preregistration": str(PREREG),
        "p": P,
        "heads": HEADS,
        "params": model.n_params_per_network(),
        "population": fitted,
        "epochs": EPOCHS,
        "seconds": result.seconds,
        "coverage": coverage,
        "n_exact": n_exact,
        "frac_exact": frac,
        "frac_exact_lo": lo,
        "frac_exact_hi": hi,
        "gcd_hist_sorted": gcd_hist,
        "n_candidate_triple": n_candidate,
        "n_certified": n_certified,
        "n_with_faithful_head": n_faithful_head,
        "score": score,
        "breaks": {},
        "certified_nets": [c for c in certs if c["certified"]][:16],
    }
    breaks: dict[str, int] = {}
    for c in certs:
        if not c["certified"]:
            b = str(c.get("break_at"))
            breaks[b] = breaks.get(b, 0) + 1
    payload["breaks"] = breaks
    write_json(parent / "metrics" / "three_torus.json", payload)
    write_json(sub / "metrics" / "three_torus.json", payload)
    print(json.dumps({k: payload[k] for k in (
        "n_exact", "n_candidate_triple", "n_certified", "n_with_faithful_head",
        "gcd_hist_sorted", "breaks", "score", "seconds",
    )}, indent=2))
    print("SCIENTIFIC 3-TORUS:", score["scientific_3_torus"])
    print("wrote", parent)


if __name__ == "__main__":
    main()
