#!/usr/bin/env python3
"""Level 5 Stage A: H=1 Model C held-row on p=7,11,17, plus existing p=13.

Predictions P1–P7 are written to disk before any new training.
Does not run Stage B (larger primes), C (scrambled), or D (sparsity).
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.analysis.cyclic import PREDICTIONS, STAGE_A_PRIMES, all_cards, prime_card
from src.analysis.group_law import load_population
from src.analysis.level4_h1 import analyze_failures_h1, analyze_h1_exact
from src.analysis.stats import wilson_interval
from src.config import ExperimentConfig
from src.hardware import resolve_device
from src.repro import environment_record, seed_everything
from src.runner import finalize_population, fit_and_build, prepare_task, train_prepared
from src.storage import new_run_dir, write_json
from src.tasks.base import SplitKind
from src.tasks.modular import ModularMultiplicationStar
from src.visualization.stage_a import (
    plot_exact_vs_p,
    plot_families_vs_phi,
    plot_hom_vs_p,
    plot_params_vs_p,
    plot_symbolic_vs_p,
    plot_table_vs_dim,
    plot_winding_histograms,
)

P13_HELD_ROW = ROOT / "results" / "runs" / "20260909_104844_held_row"
TRAIN_PRIMES = (7, 11, 17)
MODEL_EQUATIONS = {
    "embedding": "e(a) = (x_a, y_a) in R^2, shared across slots, Xavier on last two dims, no dlog init",
    "interaction": "h = (x_a x_b - y_a y_b,  x_a y_b + y_a x_b)",
    "readout": "logits = h @ W + b,  W in R^{2 x (p-1)}, b in R^{p-1}",
    "params": "5(p-1) = 2(p-1) embeddings + 2(p-1) readout + (p-1) bias",
    "latent_dim": 2,
}


def held_row_coverage(p: int, a0: int) -> dict:
    task = ModularMultiplicationStar(p=p)
    a, b, _, y = task.full_domain("cpu")
    split = task.split(SplitKind.HELD_ROW, seed=0, held_operands=[a0])
    train_a, train_b = a[split.train_idx], b[split.train_idx]
    test_a, test_b = a[split.test_idx], b[split.test_idx]
    train_y, test_y = y[split.train_idx], y[split.test_idx]
    residues = set(range(1, p))
    return {
        "p": p,
        "a0": a0,
        "n_train": int(split.train_idx.numel()),
        "n_test": int(split.test_idx.numel()),
        "n_domain": task.n_domain,
        "expected_train": (p - 2) * (p - 1),
        "expected_test": p - 1,
        "train_has_all_residues_as_a_or_b": residues <= set(train_a.tolist()) | set(train_b.tolist()),
        "a0_appears_as_b_in_train": a0 in set(int(v) for v in train_b.tolist()),
        "a0_never_as_a_in_train": a0 not in set(int(v) for v in train_a.tolist()),
        "n_train_classes": int(train_y.unique().numel()),
        "n_test_classes": int(test_y.unique().numel()),
        "all_classes_in_train": int(train_y.unique().numel()) == p - 1,
        "all_classes_in_test": int(test_y.unique().numel()) == p - 1,
        "every_residue_in_train_as_b": residues <= set(int(v) for v in train_b.tolist()),
    }


def prerun_document() -> dict:
    cards = all_cards(STAGE_A_PRIMES)
    coverage = [held_row_coverage(c["p"], c["held_residue"]) for c in cards]
    p13_seconds = 31.1
    runtime = []
    for c in cards:
        scale = c["n_domain"] / 144.0
        runtime.append(
            {
                "p": c["p"],
                "est_seconds_vs_p13": None if c["p"] == 13 else round(p13_seconds * scale, 1),
                "domain_vs_p13": round(scale, 3),
                "params": c["params_h1"],
                "params_vs_p13": round(c["params_h1"] / 60.0, 3),
            }
        )
    return {
        "stage": "A",
        "predictions_locked": PREDICTIONS,
        "architecture": MODEL_EQUATIONS,
        "parameter_count": "5(p-1); embedding dim stays 2; total params grow with p",
        "prime_cards": cards,
        "held_row_coverage": coverage,
        "winding_fit": (
            "Least-squares scaled/rotated/conjugated dlog circle over k=0..p-2. "
            "k_eff = k if not conjugate else (-k) mod (p-1). "
            "Homomorphism: wrap(theta(ab)-theta(a)-theta(b)), then subtract circular mean."
        ),
        "scrambled_table_plan": (
            "Stage C, not run. Shuffle labels of the (p-1)^2 table while preserving "
            "class counts (each residue appears p-1 times). This keeps marginals and "
            "held-row geometry but is not a group homomorphism, so it should break "
            "associativity. Do not use y=pi(ab), which is isomorphic to F_p*."
        ),
        "runtime_estimate": {
            "p13_h1_held_row_seconds": p13_seconds,
            "population": 1024,
            "epochs": 8000,
            "note": "p=17 domain is 256/144 ≈ 1.78×; three new primes should be a few minutes on the same GPU.",
            "per_prime": runtime,
        },
        "code_changes": [
            "p is a task config field; held_operands [-1] maps to p-1",
            "winding search includes k=0",
            "analysis is parameterized by p rather than hardcoded 13",
            "exact nets are analyzed live (not capped at 256) for new primes",
        ],
        "not_in_this_stage": ["Stage B p=19,23,29,31", "Stage C scrambled", "Stage D sparsity", "Model D", "Collatz"],
    }


def _summary_row(p: int, train_row: dict, exact: dict, source: str) -> dict:
    card = prime_card(p)
    n_pop = int(train_row["population"])
    n_exact = int(train_row["n_exact"])
    frac, lo, hi = wilson_interval(n_exact, n_pop)
    analyzed = int(exact.get("n_analyzed", 0))
    hom = exact.get("hom") or {}
    fit = exact.get("fit") or {}
    return {
        "p": p,
        "n": card["n"],
        "g": card["primitive_root"],
        "phi": card["phi"],
        "units": card["units"],
        "frac_k_units": card["frac_k_units"],
        "p_minus_1_factorization": card["p_minus_1_factorization"],
        "train_pairs": train_row["n_train"],
        "test_pairs": train_row["n_test"],
        "n_domain": train_row["n_domain"],
        "population": n_pop,
        "exact_count": n_exact,
        "exact_fraction": frac,
        "wilson_lo": lo,
        "wilson_hi": hi,
        "params": train_row["params"],
        "params_per_element": train_row["params"] / card["n"],
        "observed_successful_k": exact.get("fit", {}).get("observed_units", []) if analyzed else [],
        "observed_non_unit_k": exact.get("fit", {}).get("observed_non_units", []) if analyzed else [],
        "fraction_unit_k": fit.get("frac_unit_k") if analyzed else None,
        "n_observed_faithful_families": exact.get("n_observed_faithful_families", 0) if analyzed else 0,
        "mean_hom_deg": (exact.get("unit_exact") or {}).get("mean_abs_centered_deg") if analyzed else None,
        "mean_hom_deg_all_exact": hom.get("mean_abs_centered_deg") if analyzed else None,
        "held_row_hom_deg": (exact.get("unit_exact") or {}).get("mean_abs_centered_held_deg") if analyzed else None,
        "train_hom_deg": (exact.get("unit_exact") or {}).get("mean_abs_centered_train_deg") if analyzed else None,
        "symbolic_exact_fraction": (exact.get("unit_exact") or {}).get("frac_symbolic_exact") if analyzed else None,
        "symbolic_exact_fraction_all": exact.get("frac_symbolic_program_exact") if analyzed else None,
        "frac_A_still_exact": exact.get("frac_A_still_exact") if analyzed else None,
        "n_unit_exact": (exact.get("unit_exact") or {}).get("n") if analyzed else 0,
        "n_nonunit_exact": (exact.get("nonunit_exact") or {}).get("n") if analyzed else 0,
        "n_analyzed": analyzed,
        "level4_replication": bool(exact.get("level4_mechanism_present")) if analyzed else False,
        "level4_exclusive_units": bool(exact.get("level4_exclusive_units")) if analyzed else False,
        "winding_table": exact.get("winding_table", []),
        "source": source,
        "seconds": train_row.get("seconds"),
        "coverage": train_row.get("coverage"),
    }


def run_prime(cfg: ExperimentConfig, p: int, device, parent: Path) -> dict:
    local = copy.deepcopy(cfg)
    local.task.p = p
    local.task.split = "held_row"
    local.task.held_operands = [-1]
    local.model.interaction = "complex"
    local.model.hidden_dims = [1]
    local.model.activation = "identity"
    local.run_name = f"complex_h1_p{p}_held_row"
    a0 = p - 1
    coverage = held_row_coverage(p, a0)
    print(
        f"\n=== p={p} coverage a0={a0} train={coverage['n_train']} "
        f"(expected {coverage['expected_train']}) test={coverage['n_test']} "
        f"train_classes={coverage['n_train_classes']}/{p-1} "
        f"a0_as_b={coverage['a0_appears_as_b_in_train']} "
        f"all_residues_trained={coverage['train_has_all_residues_as_a_or_b']} ==="
    )
    if not coverage["all_classes_in_train"] or not coverage["a0_appears_as_b_in_train"]:
        raise RuntimeError(f"held-row split failed coverage checks for p={p}: {coverage}")

    seed_everything(cfg.seed)
    prepared = prepare_task(local, device, protocol="split")
    model, fitted = fit_and_build(local, prepared, device, seed=cfg.seed)
    sub = parent / local.run_name
    for d in ("metrics", "checkpoints", "figures", "logs"):
        (sub / d).mkdir(parents=True, exist_ok=True)
    print(
        f"p={p}: P={fitted} params={model.n_params_per_network()} "
        f"g={prime_card(p)['primitive_root']} phi={prime_card(p)['phi']}"
    )
    result = train_prepared(model, prepared, local, show_progress=True)
    packed = finalize_population(
        model, result, prepared, sub, write_figures=True, max_exact_saved=fitted
    )
    counts = packed["counts"]
    frac, lo, hi = wilson_interval(counts.exact, counts.total)
    exact_mask = packed["labels"] == 4
    exact = analyze_h1_exact(
        model,
        exact_mask,
        prepared.domain_x,
        prepared.domain_y,
        prepared.split.train_idx,
        prepared.split.test_idx,
        p,
        a0,
    )
    failures = analyze_failures_h1(model, exact_mask, p)
    train_row = {
        "p": p,
        "n_train": int(prepared.train_y.numel()),
        "n_test": int(prepared.test_y.numel()),
        "n_domain": prepared.task.n_domain,
        "params": model.n_params_per_network(),
        "population": fitted,
        "n_exact": counts.exact,
        "frac_exact": frac,
        "frac_exact_lo": lo,
        "frac_exact_hi": hi,
        "seconds": result.seconds,
        "mean_train_acc": float(result.final_train_acc.mean()),
        "mean_test_acc": float(result.final_test_acc.mean()),
        "mean_domain_acc": float(result.final_domain_acc.mean()),
        "counts": counts.as_dict(),
        "coverage": coverage,
        "split_info": prepared.split.info,
    }
    payload = {
        "train": train_row,
        "exact": exact,
        "failures": failures,
        "summary": _summary_row(p, train_row, exact, source="this_run"),
    }
    write_json(sub / "metrics" / "stage_a_prime.json", payload)
    print(
        f"  exact {counts.exact}/{counts.total} ({frac:.4f}) [{lo:.3f},{hi:.3f}]  "
        f"unit_k={exact.get('fit', {}).get('frac_unit_k')}  "
        f"hom={exact.get('hom', {}).get('mean_abs_centered_deg')}  "
        f"symbolic={exact.get('frac_symbolic_program_exact')}  "
        f"L4={exact.get('level4_replication')}  {result.seconds:.1f}s"
    )
    if torch.cuda.is_available():
        del model
        torch.cuda.empty_cache()
    return payload


def attach_p13(device) -> dict:
    row_path = P13_HELD_ROW / "complex_h1_held_row" / "metrics" / "row.json"
    ckpt_path = P13_HELD_ROW / "complex_h1_held_row" / "checkpoints" / "successful.pt"
    fail_path = P13_HELD_ROW / "complex_h1_held_row" / "checkpoints" / "representatives.pt"
    with open(row_path, encoding="utf-8") as f:
        train_row = json.load(f)
    train_row = {
        "p": 13,
        "n_train": train_row["n_train"],
        "n_test": train_row["n_test"],
        "n_domain": train_row["n_domain"],
        "params": train_row["params"],
        "population": train_row["population"],
        "n_exact": train_row["n_exact"],
        "frac_exact": train_row["frac_exact"],
        "frac_exact_lo": train_row["frac_exact_lo"],
        "frac_exact_hi": train_row["frac_exact_hi"],
        "seconds": train_row["seconds"],
        "mean_train_acc": train_row["mean_train_acc"],
        "mean_test_acc": train_row["mean_test_acc"],
        "mean_domain_acc": train_row["mean_domain_acc"],
        "counts": train_row["counts"],
        "coverage": held_row_coverage(13, 12),
        "split_info": train_row["info"],
    }
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = load_population(ckpt, device)
    task = ModularMultiplicationStar(p=13)
    _, _, x, y = task.full_domain(device)
    split = task.split(SplitKind.HELD_ROW, seed=0, held_operands=[12])
    exact_mask = torch.ones(model.population, dtype=torch.bool)
    exact = analyze_h1_exact(model, exact_mask, x, y, split.train_idx, split.test_idx, 13, 12)
    exact["n_exact_population"] = train_row["n_exact"]
    exact["note"] = (
        f"winding/hom/symbolic measured on {exact['n_analyzed']} saved exact nets "
        f"of {train_row['n_exact']} population exact (finalize cap was 256)"
    )
    failures = {"n_fail": None, "note": "full failure population not saved; see representatives"}
    if fail_path.exists():
        fckpt = torch.load(fail_path, map_location="cpu", weights_only=False)
        labels = fckpt.get("labels")
        if labels is not None:
            fmodel = load_population(fckpt, device)
            fail_mask = labels.cpu() == 4
            failures = analyze_failures_h1(fmodel, fail_mask.cpu(), 13)
            failures["note"] = "representatives only, not the full failed population"
    payload = {
        "train": train_row,
        "exact": exact,
        "failures": failures,
        "summary": _summary_row(13, train_row, exact, source=str(P13_HELD_ROW)),
    }
    return payload


def reanalyze_saved_prime(run_dir: Path, p: int, device) -> dict:
    sub = run_dir / f"complex_h1_p{p}_held_row"
    with open(sub / "metrics" / "stage_a_prime.json", encoding="utf-8") as f:
        prev = json.load(f)
    ckpt = torch.load(sub / "checkpoints" / "successful.pt", map_location="cpu", weights_only=False)
    model = load_population(ckpt, device)
    task = ModularMultiplicationStar(p=p)
    _, _, x, y = task.full_domain(device)
    a0 = p - 1
    split = task.split(SplitKind.HELD_ROW, seed=0, held_operands=[a0])
    exact_mask = torch.ones(model.population, dtype=torch.bool)
    exact = analyze_h1_exact(model, exact_mask, x, y, split.train_idx, split.test_idx, p, a0)
    payload = {
        "train": prev["train"],
        "exact": exact,
        "failures": prev["failures"],
        "summary": _summary_row(p, prev["train"], exact, source="this_run"),
    }
    write_json(sub / "metrics" / "stage_a_prime.json", payload)
    print(
        f"reanalyzed p={p}: unit_exact={exact.get('unit_exact', {}).get('n')} "
        f"nonunit={exact.get('nonunit_exact', {}).get('n')} "
        f"unit_hom={exact.get('unit_exact', {}).get('mean_abs_centered_deg')} "
        f"symbolic|unit={exact.get('unit_exact', {}).get('frac_symbolic_exact')} "
        f"L4mech={exact.get('level4_mechanism_present')} exclusive={exact.get('level4_exclusive_units')}"
    )
    return payload


def write_stage_a_report(run_dir: Path, by_p: dict) -> None:
    summaries = [
        _summary_row(p, by_p[p]["train"], by_p[p]["exact"], by_p[p]["summary"]["source"])
        for p in STAGE_A_PRIMES
    ]
    n_l4 = sum(1 for r in summaries if r["level4_replication"])
    n_excl = sum(1 for r in summaries if r["level4_exclusive_units"])
    payload = {
        "stage": "A",
        "stop": "Do not proceed to Stage B until this report is read.",
        "predictions": PREDICTIONS,
        "n_level4_replications": n_l4,
        "n_exclusive_unit_primes": n_excl,
        "level5": False,
        "level5_reason": "Level 5 requires preferably ≥5 distinct primes with Level-4 replications. Stage A has at most 4.",
        "rows": summaries,
        "failures": {str(p): by_p[p]["failures"] for p in STAGE_A_PRIMES},
        "exact": {str(p): by_p[p]["exact"] for p in STAGE_A_PRIMES},
    }
    write_json(run_dir / "metrics" / "stage_a.json", payload)
    figs = run_dir / "figures"
    plot_exact_vs_p(summaries, figs / "exact_vs_p.png", "p", "p", "P(exact generalizer) vs p")
    plot_exact_vs_p(summaries, figs / "exact_vs_order.png", "n", "p−1 (group order)", "P(exact generalizer) vs group order")
    plot_families_vs_phi(summaries, figs / "families_vs_phi.png")
    plot_hom_vs_p([r for r in summaries if r["mean_hom_deg"] is not None], figs / "hom_vs_p.png")
    plot_symbolic_vs_p([r for r in summaries if r["symbolic_exact_fraction"] is not None], figs / "symbolic_vs_p.png")
    plot_table_vs_dim(summaries, figs / "table_vs_dim.png")
    plot_params_vs_p(summaries, figs / "params_vs_p.png")
    plot_winding_histograms(summaries, figs / "winding_histograms.png")
    print("\n===== Stage A summary =====")
    print(
        "p  p-1  g  phi  train  test  P  exact  frac  Wilson  k_units  frac_unit  "
        "hom_unit  held_hom  symbolic|unit  L4  exclusive"
    )
    for r in summaries:
        print(
            f"{r['p']:2d} {r['n']:3d} {r['g']:2d} {r['phi']:3d} {r['train_pairs']:5d} {r['test_pairs']:4d} "
            f"{r['population']:4d} {r['exact_count']:5d} {r['exact_fraction']:.4f} "
            f"[{r['wilson_lo']:.3f},{r['wilson_hi']:.3f}] {r['observed_successful_k']} "
            f"{r['fraction_unit_k']} {r['mean_hom_deg']} {r['held_row_hom_deg']} "
            f"{r['symbolic_exact_fraction']} {r['level4_replication']} {r['level4_exclusive_units']}"
        )
    print(f"Level-4 mechanism replications: {n_l4}/{len(summaries)}")
    print(f"Exclusive unit-k among exact: {n_excl}/{len(summaries)}")
    print("Level 5: not claimed (Stage A only).")
    print("wrote", run_dir / "metrics" / "stage_a.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "stage_a.yaml")
    parser.add_argument("--population", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--finish-run", type=Path, default=None, help="Re-analyze an existing Stage A run; skip training.")
    args = parser.parse_args()
    cfg = ExperimentConfig.from_yaml(args.config)
    if args.population:
        cfg.population.size = args.population
    if args.epochs:
        cfg.train.epochs = args.epochs
    device = resolve_device(cfg.device)

    if args.finish_run is not None:
        run_dir = args.finish_run
        print(f"finishing existing run: {run_dir}")
        by_p = {}
        for p in TRAIN_PRIMES:
            by_p[p] = reanalyze_saved_prime(run_dir, p, device)
        print("\n=== attaching existing p=13 held-row result ===")
        by_p[13] = attach_p13(device)
        write_json(run_dir / "metrics" / "p13_attached.json", by_p[13])
        write_stage_a_report(run_dir, by_p)
        return

    run_dir = new_run_dir(ROOT, cfg.run_name)
    prerun = prerun_document()
    prerun["environment"] = environment_record(str(ROOT))
    write_json(run_dir / "metrics" / "predictions.json", prerun)
    write_json(run_dir / "meta.json", {**environment_record(str(ROOT)), "stage": "A", "predictions_locked": True})
    print(f"run directory: {run_dir}")
    print("Predictions P1–P7 locked before training.")
    for k, v in PREDICTIONS.items():
        print(f"  {k}. {v}")
    print("Prime cards:")
    for c in prerun["prime_cards"]:
        print(
            f"  p={c['p']} n={c['n']} g={c['primitive_root']} phi={c['phi']} "
            f"units={c['units']} params={c['params_h1']} train={c['n_train_held_row']}"
        )
    for cov in prerun["held_row_coverage"]:
        print(
            f"  coverage p={cov['p']}: classes_train={cov['all_classes_in_train']} "
            f"a0_as_b={cov['a0_appears_as_b_in_train']} "
            f"all_residues={cov['train_has_all_residues_as_a_or_b']}"
        )

    by_p = {}
    for p in TRAIN_PRIMES:
        by_p[p] = run_prime(cfg, p, device, run_dir)
        write_json(run_dir / "metrics" / "stage_a.json", {"primes": {str(k): v["summary"] for k, v in by_p.items()}})

    print("\n=== attaching existing p=13 held-row result ===")
    by_p[13] = attach_p13(device)
    write_json(run_dir / "metrics" / "p13_attached.json", by_p[13])
    write_stage_a_report(run_dir, by_p)


if __name__ == "__main__":
    main()
