"""Model K on spanning-tree counts: preregistered Kirchhoff certificate.

Fixed cofactor of a learned weighted Laplacian, n=5, held edge e=(0,1).
Does not raise n, add a recursive DC net, 0-1-init, or loosen exactness
if n_certified=0. Does not train Collatz or Model C.
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

from src.analysis.matrix_tree import decoder_sanity, certify_net
from src.analysis.stats import wilson_interval
from src.config import ExperimentConfig
from src.hardware import resolve_device
from src.population.kirchhoff import PopulationKirchhoff
from src.repro import environment_record, seed_everything
from src.runner import finalize_population, fit_and_build, prepare_from_task, train_prepared
from src.storage import new_run_dir, write_json
from src.tasks.spanning import (
    DESTROY_SEED,
    N_DEFAULT,
    spanning_tau,
    spanning_tau_destroyed,
    held_edge_coverage,
)


PREREG = ROOT / "tiny_tao_results" / "preregistration_spanning_trees_n5.json"
POPULATION = 1024
EPOCHS = 8000
LR = 3e-3
SEED = 0
N = N_DEFAULT
HELD_EDGE = 0


def make_cfg(task_name: str, interaction: str) -> ExperimentConfig:
    cfg = ExperimentConfig.from_yaml(ROOT / "configs" / "spanning_trees_n5.yaml")
    cfg.task.name = task_name
    cfg.task.p = N
    cfg.task.split = "held_edge"
    cfg.task.held_operands = [HELD_EDGE]
    cfg.model.interaction = interaction
    if interaction == "kirchhoff":
        cfg.model.hidden_dims = [10]
        cfg.model.activation = "identity"
    else:
        cfg.model.hidden_dims = [8]
        cfg.model.activation = "relu"
    cfg.population.size = POPULATION
    cfg.train.epochs = EPOCHS
    cfg.train.lr = LR
    cfg.seed = SEED
    cfg.run_name = f"{interaction}_{task_name}_held_edge"
    return cfg


def score_kirchhoff(model: PopulationKirchhoff, exact_idx: torch.Tensor, x: np.ndarray, tau: np.ndarray) -> dict:
    if exact_idx.numel() == 0:
        return {
            "n_exact": 0,
            "n_certified": 0,
            "breaks": {},
            "certified_nets": [],
        }
    scale = model.scale.detach().cpu()[exact_idx].numpy().astype(np.float64)
    bias = model.bias.detach().cpu()[exact_idx].numpy().astype(np.float64)
    certs = []
    for i in range(scale.shape[0]):
        cert = certify_net(
            scale[i],
            bias[i],
            x,
            tau,
            model.elist,
            model.n_vertices,
            scramble_seed=int(exact_idx[i]),
        )
        cert["net"] = int(exact_idx[i])
        certs.append(cert)
    n_certified = int(sum(1 for c in certs if c["certified"]))
    breaks: dict[str, int] = {}
    for c in certs:
        if not c["certified"]:
            b = str(c.get("break_at"))
            breaks[b] = breaks.get(b, 0) + 1
    return {
        "n_exact": int(exact_idx.numel()),
        "n_certified": n_certified,
        "breaks": breaks,
        "certified_nets": [c for c in certs if c["certified"]][:16],
    }


def synthetic_forward_exact(task, device: torch.device) -> bool:
    _, _, x, y = task.full_domain(device)
    model = PopulationKirchhoff(
        population=1,
        input_dim=task.input_dim,
        hidden_dims=[task.n_edges],
        output_dim=task.output_dim,
        n_vertices=task.n,
        elist=task.elist,
        device=device,
    )
    with torch.no_grad():
        model.scale.fill_(1.0)
        model.bias.zero_()
        pred = model(x).argmax(dim=-1).squeeze(1)
        return bool((pred == y).all().item())


def run_job(task, cfg: ExperimentConfig, device: torch.device, parent: Path, certify: bool) -> dict:
    coverage = held_edge_coverage(task, HELD_EDGE)
    if task.name == "spanning_tau":
        if not coverage["complete_graph_class_test_only"] or not coverage["train_sees_zero"]:
            raise RuntimeError(coverage)
    elif coverage["n_train"] != 512 or coverage["n_test"] != 512:
        raise RuntimeError(coverage)
    seed_everything(SEED)
    prepared = prepare_from_task(task, cfg, device, protocol="split")
    model, fitted = fit_and_build(cfg, prepared, device, seed=SEED)
    sub = parent / cfg.run_name
    for d in ("metrics", "checkpoints", "figures", "logs"):
        (sub / d).mkdir(parents=True, exist_ok=True)
    print(
        f"{cfg.model.interaction} {task.name} n={N} P={fitted} "
        f"params={model.n_params_per_network()} device={device}",
        flush=True,
    )
    result = train_prepared(model, prepared, cfg, show_progress=True)
    packed = finalize_population(
        model, result, prepared, sub, write_figures=True, max_exact_saved=fitted
    )
    exact_idx = torch.where(packed["labels"].detach().cpu() == 4)[0]
    _, _, x, y = task.full_domain("cpu")
    xd = x.numpy().astype(np.float64)
    tau = y.numpy().astype(np.int64)
    if certify:
        scored = score_kirchhoff(model, exact_idx, xd, tau)
    else:
        scored = {
            "n_exact": int(exact_idx.numel()),
            "n_certified": 0,
            "breaks": {"no_cofactor": int(exact_idx.numel())},
            "certified_nets": [],
        }
    frac, lo, hi = wilson_interval(scored["n_exact"], fitted)
    payload = {
        "task": task.name,
        "interaction": cfg.model.interaction,
        "n": N,
        "held_edge": HELD_EDGE,
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
    write_json(sub / "metrics" / "kirchhoff.json", payload)
    print(
        json.dumps(
            {k: payload[k] for k in ("task", "interaction", "n_exact", "n_certified", "breaks", "seconds")},
            indent=2,
        ),
        flush=True,
    )
    return payload


def main() -> None:
    pre = json.loads(PREREG.read_text())
    assert pre["status"] == "preregistered"
    sanity = decoder_sanity()
    print(
        "synthetic decoder meet",
        sanity["meet"]["certified"],
        "destroyed",
        sanity["destroyed"]["certified"],
        "dc",
        sanity["oracle"]["dc_equals_kirchhoff"],
        "passed",
        sanity["passed"],
        flush=True,
    )
    if not sanity["passed"]:
        raise RuntimeError(f"decoder sanity failed before training: {sanity}")

    device = resolve_device("cuda")
    genuine = spanning_tau(N)
    if not synthetic_forward_exact(genuine, device):
        raise RuntimeError("scale=1,bias=0 Model K is not domain-exact before SGD")

    parent = new_run_dir(ROOT, "spanning_trees_n5")
    write_json(parent / "predictions.json", pre)
    write_json(
        parent / "decoder_sanity.json",
        {
            "passed": sanity["passed"],
            "meet_certified": sanity["meet"]["certified"],
            "destroyed_certified": sanity["destroyed"]["certified"],
            "destroyed_equals_tau": sanity["destroyed_equals_tau"],
            "oracle": sanity["oracle"],
            "destroy_seed": DESTROY_SEED,
        },
    )
    write_json(parent / "environment.json", environment_record())

    k_meet = run_job(genuine, make_cfg("spanning_tau", "kirchhoff"), device, parent, certify=True)
    k_destroyed = run_job(
        spanning_tau_destroyed(N, DESTROY_SEED),
        make_cfg("spanning_tau_destroyed", "kirchhoff"),
        device,
        parent,
        certify=True,
    )
    a_meet = run_job(spanning_tau(N), make_cfg("spanning_tau", "add"), device, parent, certify=False)

    p5_vacuous = k_meet["n_certified"] == 0
    score = {
        "P1_K_exact_exists": {
            "verdict": "hold" if k_meet["n_exact"] >= 1 else "miss",
            "n_exact": k_meet["n_exact"],
        },
        "P2_K_certified_not_guaranteed": {
            "verdict": "hold",
            "n_certified": k_meet["n_certified"],
            "note": "Appeared" if k_meet["n_certified"] else "Did not appear; valid miss",
        },
        "P3_destroyed_zero_certified": {
            "verdict": "hold" if k_destroyed["n_certified"] == 0 else "miss",
            "n_certified": k_destroyed["n_certified"],
            "n_exact": k_destroyed["n_exact"],
        },
        "P4_A_not_kirchhoff": {
            "verdict": "hold" if a_meet["n_certified"] == 0 else "miss",
            "n_certified": a_meet["n_certified"],
            "n_exact": a_meet["n_exact"],
        },
        "P5_do_not_raise_n": {
            "verdict": "not_triggered" if k_meet["n_certified"] else "applied",
            "note": "n_certified=0: do not raise n, add a recursive DC net, 0-1-init, or loosen exactness",
        },
        "P6_not_collatz": {"verdict": "applied"},
        "P7_not_model_c": {"verdict": "applied"},
    }
    scientific = k_meet["n_certified"] > 0 and k_destroyed["n_certified"] == 0
    scored = {
        "status": "scored",
        "preregistration": str(PREREG),
        "run": str(parent),
        "scored_after_training": True,
        "protocol": (
            f"Model K n={N}, held_edge={HELD_EDGE}, P={POPULATION}, "
            f"{EPOCHS} Adam steps, lr={LR}, seed={SEED}, Xavier scale, bias 0"
        ),
        "params_K": 20,
        "k_meet": k_meet,
        "k_destroyed": k_destroyed,
        "a_meet": a_meet,
        "predictions": score,
        "scientific_kirchhoff": scientific,
        "p5_vacuous": p5_vacuous,
        "lesson": (
            "Model K is a Laplacian cofactor of learned edge weights, not a character. "
            "Certified means binarized weights recover τ, drop-one-edge fails, adjacency minor fails. "
            "Destroyed-law control must not certify. Do not raise n."
        ),
    }
    write_json(parent / "metrics" / "kirchhoff.json", scored)
    out_score = ROOT / "tiny_tao_results" / "preregistration_spanning_trees_n5.scored.json"
    write_json(out_score, scored)
    print("SCIENTIFIC KIRCHHOFF:", scientific)
    print("wrote", parent)
    print("scored", out_score)


if __name__ == "__main__":
    main()
