#!/usr/bin/env python3
"""Train the locked Stage B control matrix. Does not change Model C or exactness."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.analysis.control_tables import PRIMARY_POP
from src.analysis.stage_b_classify import (
    DESTROYED_LAW,
    baseline_inconsistent,
    classify_exact_embeddings,
    family_summary,
)
from src.analysis.stats import wilson_interval
from src.config import ExperimentConfig
from src.hardware import resolve_device
from src.repro import environment_record, seed_everything
from src.runner import finalize_population, fit_and_build, prepare_from_task, train_prepared
from src.storage import write_json
from src.tasks.modular import CayleyTableStar
from src.visualization.stage_b import (
    plot_dose_response,
    plot_embedding,
    plot_hom_residuals,
    plot_rate_by_control,
    plot_solver_vs_associativity,
    plot_train_vs_held,
)

PRIORITY = {
    "genuine": 0,
    "A_isomorphic": 1,
    "D_random_comm": 2,
    "E_latin_nonassoc": 3,
    "E_latin_comm_nonassoc": 4,
    "F_assoc_damage": 5,
    "C_slot_relabel": 6,
    "G_noncyclic_group": 7,
    "B_output_perm": 8,
}

DEFAULT_RUN = ROOT / "results" / "runs" / "20260909_125737_stage_b_controls"


def load_jobs(run_dir: Path) -> list[dict]:
    pred = json.loads((run_dir / "control_predictions.json").read_text())
    jobs = [j for j in pred["planned_jobs"] if not j.get("skipped")]
    jobs.sort(key=lambda j: (PRIORITY.get(j["control"], 99), j["p"], j.get("frac") or 0, j["instance"]))
    return jobs


def load_table(run_dir: Path, job_id: str) -> tuple[np.ndarray, dict, np.ndarray | None]:
    T = np.load(run_dir / "tables" / f"{job_id}.npy")
    meta = json.loads((run_dir / "tables" / f"{job_id}.json").read_text())
    pi_path = run_dir / "tables" / f"{job_id}_pi.npy"
    pi = np.load(pi_path) if pi_path.exists() else None
    return T, meta, pi


def job_dir(run_dir: Path, job_id: str) -> Path:
    d = run_dir / "jobs" / job_id
    for sub in ("metrics", "checkpoints", "figures", "logs"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    return d


def already_done(run_dir: Path, job_id: str) -> bool:
    return (run_dir / "jobs" / job_id / "metrics" / "job.json").exists()


def run_job(cfg: ExperimentConfig, job: dict, run_dir: Path, device: torch.device) -> dict:
    job_id = job["job_id"]
    T, meta, pi = load_table(run_dir, job_id)
    p = int(job["p"])
    local = ExperimentConfig.from_yaml(ROOT / "configs" / "stage_b_controls.yaml")
    local.task.p = p
    local.task.split = "held_row"
    local.task.held_operands = [-1]
    local.model.interaction = "complex"
    local.model.hidden_dims = [1]
    local.model.activation = "identity"
    local.population.size = int(job["population"])
    local.train.epochs = int(cfg.train.epochs)
    local.train.lr = float(cfg.train.lr)
    local.seed = 0
    local.run_name = job_id

    task = CayleyTableStar(table=T, p=p, name=f"cayley_{job_id}")
    seed_everything(local.seed)
    prepared = prepare_from_task(task, local, device, protocol="split")
    train_classes = int(torch.unique(prepared.train_y).numel())
    if train_classes != p - 1:
        raise RuntimeError(f"{job_id}: train missing classes ({train_classes}/{p-1})")

    model, fitted = fit_and_build(local, prepared, device, seed=local.seed)
    sub = job_dir(run_dir, job_id)
    print(
        f"\n=== {job_id}  p={p} pop={fitted}  A={meta['diagnostics']['associativity_fraction']:.4f} "
        f"latin={meta['diagnostics']['latin']} group={meta['diagnostics']['is_group']} ===",
        flush=True,
    )
    result = train_prepared(model, prepared, local, show_progress=True)
    packed = finalize_population(
        model, result, prepared, sub, write_figures=False, max_exact_saved=fitted
    )
    counts = packed["counts"]
    exact_idx = torch.where(packed["labels"] == 4)[0]
    xy = np.zeros((0, p - 1, 2), dtype=np.float64)
    if exact_idx.numel():
        xy = model.E.detach().cpu().numpy()[exact_idx.cpu().numpy(), :, 0, :].astype(np.float64)

    classif = classify_exact_embeddings(xy, T, p, job["control"], pi=pi)
    n_exact = int(counts.exact)
    n_pop = int(counts.total)
    pe, lo, hi = wilson_interval(n_exact, n_pop)
    row = {
        "job_id": job_id,
        "prime": p,
        "p": p,
        "control": job["control"],
        "control_type": job["control"],
        "control_instance": job["instance"],
        "instance": job["instance"],
        "frac": job.get("frac"),
        "table_order": p - 1,
        "commutative": meta["diagnostics"]["commutative"],
        "associative": meta["diagnostics"]["associative"],
        "associativity_fraction": meta["diagnostics"]["associativity_fraction"],
        "latin": meta["diagnostics"]["latin"],
        "identity": meta["diagnostics"]["identity_exists"],
        "population": n_pop,
        "n_exact": n_exact,
        "n_faithful": classif["n_faithful"],
        "n_class2": classif["n_class2"],
        "n_other_symbolic": classif["n_other_symbolic"],
        "frac_exact": n_exact / n_pop,
        "frac_faithful": classif["n_faithful"] / n_pop,
        "frac_class2": classif["n_class2"] / n_pop,
        "frac_exact_lo": lo,
        "frac_exact_hi": hi,
        "mean_train_acc": float(result.final_train_acc.mean()),
        "mean_test_acc": float(result.final_test_acc.mean()),
        "mean_domain_acc": float(result.final_domain_acc.mean()),
        "train_accuracy": float(result.final_train_acc.mean()),
        "held_row_accuracy": float(result.final_test_acc.mean()),
        "full_domain_accuracy": float(result.final_domain_acc.mean()),
        "seconds": result.seconds,
        "n_saved": classif["n_saved"],
        "classification": classif,
        "held_row": meta["held_row"],
        "role": job.get("role"),
    }
    write_json(sub / "metrics" / "job.json", row)
    print(
        f"  exact {n_exact}/{n_pop}  F={classif['n_faithful']}  Q={classif['n_class2']}  "
        f"other={classif['n_other_symbolic']}  train={row['mean_train_acc']:.3f}  "
        f"held={row['mean_test_acc']:.3f}  {result.seconds:.1f}s",
        flush=True,
    )
    if torch.cuda.is_available():
        del model
        torch.cuda.empty_cache()
    return row


def check_baselines(rows: list[dict], priors: dict) -> list[str]:
    msgs = []
    for r in rows:
        bad, why = baseline_inconsistent(r["p"], r["n_exact"], r["population"], r["n_class2"], priors[r["p"]])
        if bad:
            msgs.append(f"p={r['p']}: {why} (exact {r['n_exact']}/{r['population']}, Class-2 {r['n_class2']})")
    return msgs


def write_outputs(run_dir: Path, rows: list[dict], stopped_early: str | None) -> None:
    figs = run_dir / "figures"
    figs.mkdir(exist_ok=True)
    fam = family_summary(rows)
    write_json(run_dir / "metrics" / "master_table.json", rows)
    write_json(run_dir / "metrics" / "family_summary.json", fam)
    write_json(
        run_dir / "metrics" / "stage_b_report.json",
        {
            "n_jobs": len(rows),
            "stopped_early": stopped_early,
            "family_summary": fam,
            "training_status": "complete" if not stopped_early else "stopped_early",
        },
    )
    plot_rate_by_control(rows, "n_exact", "P(exact)", "Exact solver rate by control type", figs / "exact_by_control.png")
    plot_rate_by_control(rows, "n_faithful", "P(Family F)", "Faithful rate by control type", figs / "faithful_by_control.png")
    plot_rate_by_control(rows, "n_class2", "P(Family Q)", "Class-2 rate by control type", figs / "class2_by_control.png")
    plot_solver_vs_associativity(rows, figs / "solver_vs_associativity.png")
    plot_train_vs_held(rows, figs / "train_vs_held.png")
    plot_dose_response(rows, figs / "associativity_dose.png")
    plot_hom_residuals(rows, figs / "hom_residuals.png")

    # representative embeddings from primary instances
    wanted = [
        ("p11_genuine_i0", "genuine"),
        ("p19_genuine_i0", "genuine p19"),
        ("p11_A_isomorphic_i0", "isomorphic"),
        ("p19_A_isomorphic_i0", "isomorphic p19"),
        ("p11_D_random_comm_i0", "random commutative"),
        ("p11_E_latin_nonassoc_i0", "Latin nonassoc"),
    ]
    for job_id, title in wanted:
        ckpt = run_dir / "jobs" / job_id / "checkpoints" / "successful.pt"
        if not ckpt.exists():
            continue
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        E = blob["E"].numpy()
        if E.shape[0] == 0:
            continue
        plot_embedding(E[0, :, 0, :], f"{title} net0 {job_id}", figs / f"embed_{job_id}.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "stage_b_controls.yaml")
    args = parser.parse_args()
    run_dir: Path = args.run_dir
    cfg = ExperimentConfig.from_yaml(args.config)
    pred = json.loads((run_dir / "control_predictions.json").read_text())
    raw_priors = pred["baseline_replication"]["prior_counts"]
    priors = {int(k): v for k, v in raw_priors.items()}

    jobs = load_jobs(run_dir)
    device = resolve_device(cfg.device)
    print(f"Stage B train  run_dir={run_dir}  device={device}  jobs={len(jobs)}", flush=True)
    write_json(
        run_dir / "meta.json",
        {
            "stage": "B",
            "training_status": "running",
            "environment": environment_record(),
            "preregistration": str(run_dir / "control_predictions.json"),
        },
    )

    rows: list[dict] = []
    # reload completed
    for j in jobs:
        pth = run_dir / "jobs" / j["job_id"] / "metrics" / "job.json"
        if pth.exists():
            rows.append(json.loads(pth.read_text()))

    stopped_early = None
    genuine_jobs = [j for j in jobs if j["control"] == "genuine"]
    other_jobs = [j for j in jobs if j["control"] != "genuine"]

    def persist():
        write_json(run_dir / "metrics" / "train_progress.json", {"n_done": len(rows), "rows": rows})

    for j in genuine_jobs:
        if already_done(run_dir, j["job_id"]):
            print(f"skip completed {j['job_id']}", flush=True)
            continue
        rows.append(run_job(cfg, j, run_dir, device))
        persist()

    genuine_rows = [r for r in rows if r["control"] == "genuine"]
    problems = check_baselines(genuine_rows, priors)
    if problems:
        stopped_early = "baseline_inconsistent: " + " | ".join(problems)
        print("STOP: baseline grossly inconsistent with locked priors:", stopped_early, flush=True)
        write_outputs(run_dir, rows, stopped_early)
        write_json(run_dir / "meta.json", {"stage": "B", "training_status": "stopped_baseline", "why": problems})
        raise SystemExit(stopped_early)

    print("Baselines consistent with locked priors. Continuing controls.", flush=True)

    for j in other_jobs:
        if already_done(run_dir, j["job_id"]):
            print(f"skip completed {j['job_id']}", flush=True)
            continue
        row = run_job(cfg, j, run_dir, device)
        rows.append(row)
        persist()
        if (
            j["control"] in DESTROYED_LAW
            and j["instance"] == 0
            and j["population"] >= PRIMARY_POP
            and row["n_other_symbolic"] >= 16
        ):
            stopped_early = (
                f"C9 novel composition family on {j['job_id']}: "
                f"{row['n_other_symbolic']} neural-free composition-exact nets"
            )
            print("STOP EVEN EARLIER:", stopped_early, flush=True)
            break

    write_outputs(run_dir, rows, stopped_early)
    write_json(
        run_dir / "meta.json",
        {
            "stage": "B",
            "training_status": "complete" if not stopped_early else "stopped_early",
            "stopped_early": stopped_early,
            "n_jobs": len(rows),
            "environment": environment_record(),
        },
    )
    print("\n===== Stage B family summary =====")
    print(f"{'control':<28} {'jobs':>4} {'pop':>6} {'exact':>6} {'F':>5} {'Q':>5} {'other':>6} {'P(ex)':>7}")
    for f in family_summary(rows):
        print(
            f"{f['control']:<28} {f['n_jobs']:4d} {f['n_pop']:6d} {f['n_exact']:6d} "
            f"{f['n_faithful']:5d} {f['n_class2']:5d} {f['n_other_symbolic']:6d} {f['p_exact']:.4f}"
        )
    print(f"\nFigures: {run_dir / 'figures'}")
    print("STOP. No extra controls, primes, p=23, Collatz, or Model D.")


if __name__ == "__main__":
    main()
