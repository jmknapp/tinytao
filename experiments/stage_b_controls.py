#!/usr/bin/env python3
"""Stage B control experiment: group law vs table interpolation.

Default: write preregistration, generate and verify every control table, STOP.
Does not train unless --train is passed AFTER the preregistration is accepted.
Does not add primes, change Model C, relax exactness, investigate p=23, or
move to Collatz / Model D.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.analysis.control_tables import (
    CLASS2_CERTIFICATE,
    CORRUPT_FRACS,
    EPOCHS,
    FAITHFUL_CERTIFICATE,
    LR,
    MASTER_SEED,
    MAX_RETRY,
    N_INSTANCES,
    PREDICTIONS,
    PRIMARY_POP,
    PRIMES,
    REPL_POP,
    STRUCTURE_MATRIX,
    generate_verified_table,
    genuine_mul,
    planned_jobs,
)
from src.repro import environment_record
from src.storage import new_run_dir, write_json, write_yaml

PRIOR_BASELINE = {
    11: {
        "exact": 553,
        "faithful": 458,
        "class2": 95,
        "other": 0,
        "source": "results/runs/20260909_112501_stage_a_primes + Class-2 certificate",
    },
    19: {
        "exact": 409,
        "faithful": 348,
        "class2": 61,
        "other": 0,
        "source": "results/runs/20260909_123440_class2_mod4",
    },
    29: {
        "exact": 439,
        "faithful": 439,
        "class2": 0,
        "other": 0,
        "source": "results/runs/20260909_123440_class2_mod4",
    },
}

SECONDS_1024 = {11: 30.0, 19: 29.9, 29: 44.9}
SECONDS_256_SCALE = 0.5


def _fix_path() -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


def estimate_runtime(jobs: list[dict]) -> dict:
    rows = []
    total_lo = 0.0
    total_hi = 0.0
    n_train = 0
    n_skip = 0
    net_equiv = 0.0
    for j in jobs:
        if j.get("skipped"):
            n_skip += 1
            continue
        n_train += 1
        p = j["p"]
        pop = j["population"]
        t1024 = SECONDS_1024[p]
        t_lo = t1024 * (SECONDS_256_SCALE if pop < PRIMARY_POP else 1.0)
        t_hi = t1024
        total_lo += t_lo
        total_hi += t_hi
        net_equiv += pop / PRIMARY_POP
        rows.append(
            {
                "job_id": j["job_id"],
                "est_seconds_if_256_scales": round(t_lo, 1),
                "est_seconds_if_launch_bound": round(t_hi, 1),
            }
        )
    return {
        "hardware_reference": "RTX 5060 Ti, prior Stage A / H_CLASS2_MOD4 wall times",
        "seconds_per_1024_by_p": SECONDS_1024,
        "n_training_jobs": n_train,
        "n_skipped_jobs": n_skip,
        "n_1024_equivalents": round(net_equiv, 2),
        "est_minutes_if_256_scales": round(total_lo / 60.0, 1),
        "est_minutes_if_launch_bound": round(total_hi / 60.0, 1),
        "note": (
            "Vectorized populations may make 256-net jobs nearly as expensive as "
            "1024-net jobs. Report both bounds. Sequential single-GPU assumed."
        ),
    }


def preregistration_payload(run_dir: Path, jobs: list[dict]) -> dict:
    runtime = estimate_runtime(jobs)
    return {
        "stage": "B",
        "title": "Control experiment: group law vs finite-table interpolation",
        "central_question": (
            "Are these tiny networks discovering representations of the underlying "
            "algebraic group law, or can the same apparently symbolic mechanisms "
            "arise when comparable finite tables are scrambled in ways that destroy "
            "the relevant algebra?"
        ),
        "mechanisms_under_test": {
            "FAMILY_F": "faithful cyclic-character solver",
            "FAMILY_Q": "quotient + radial-lift / quadratic-character solver",
        },
        "must_distinguish": [
            "finite-table interpolation",
            "superficial symmetry exploitation",
            "low-complexity table structure",
            "commutative-table structure",
            "quasigroup/Latin-square structure",
            "genuine associative group structure",
        ],
        "labels_vs_law": {
            "scrambled_labels": (
                "Control A. Same cyclic group, random bijection π. POSITIVE control. "
                "Do not call this scrambled multiplication in the destroyed-law sense."
            ),
            "scrambled_law": (
                "Controls D, E, F. Preserve some combinatorial properties while "
                "destroying associativity / the group axioms."
            ),
        },
        "primes_locked": list(PRIMES),
        "prime_reasons": {
            "11": "original system where both faithful and Class-2 were characterized",
            "19": "prospective p≡3 (mod 4) positive Class-2 result",
            "29": "p≡1 (mod 4) comparison: many faithful exact solvers, 0 Class-2",
        },
        "p31_decision": (
            "p=31 is algebraically eligible (≡3 mod 4; 48 Class-2 / 1024 in the "
            "prospective test) but is NOT included. Decision made before table "
            "generation and before training, to bound GPU cost. Do not add p=31 "
            "or any other prime after seeing results."
        ),
        "p23_decision": (
            "p=23 is not in this experiment. Do not reinterpret the 0 exact / 1024 "
            "float32-gate result during Stage B."
        ),
        "not_in_this_stage": [
            "Collatz",
            "new primes",
            "Model C changes",
            "exactness-gate relaxation",
            "p=23 near-exact investigation",
            "Model D",
        ],
        "master_seed": MASTER_SEED,
        "seed_formula": (
            "MASTER + control_id*1_000_000 + p*10_000 + instance*1_000 + extra*100 + retry"
        ),
        "n_instances_stochastic": N_INSTANCES,
        "population_primary_instance0": PRIMARY_POP,
        "population_replication_instances_1_to_4": REPL_POP,
        "optimizer": "adam",
        "lr": LR,
        "steps": EPOCHS,
        "init": "Xavier uniform, no dlog / Legendre init",
        "architecture": "Model C H=1, shared 2-D embedding, h=z(a)z(b), linear readout",
        "held_row": {
            "rule": "hold a0 = final displayed symbol out of the first operand slot",
            "genuine_residue": "p-1",
            "index": "n-1 = p-2",
            "a0_remains_present_as_b": True,
            "regenerate_if_train_missing_a_class": True,
            "max_retry": MAX_RETRY,
        },
        "exactness_criterion": {
            "domain_acc": ">= 1.0 in float32",
            "do_not_relax": True,
            "same_as": ["Stage A", "H_CLASS2_MOD4"],
        },
        "classification_pipeline": [
            "every exact net starts as EXACT-UNCLASSIFIED",
            "attempt faithful-character certificate (Family F)",
            "attempt quotient+radial Class-2 certificate (Family Q)",
            "attempt alternative symbolic extraction",
            "do not classify by R² alone",
        ],
        "faithful_certificate": FAITHFUL_CERTIFICATE,
        "class2_certificate": CLASS2_CERTIFICATE,
        "predictions": PREDICTIONS,
        "baseline_replication": {
            "purpose": "verify current code reproduces established behavior; not a discovery run",
            "protocol": "1024 nets, 8000 Adam steps, lr=3e-3, Xavier, H=1, held-row a0=p-1",
            "prior_counts": PRIOR_BASELINE,
            "stop_if_grossly_inconsistent": {
                "exact_count_is_zero": True,
                "absolute_exact_rate_diff_gt": 0.20,
                "p11_or_p19_class2_count_is_zero": True,
                "p29_class2_count_positive": True,
                "action": "STOP and diagnose before running any non-genuine control",
            },
        },
        "control_definitions": {
            "genuine": "y(a,b) = a*b mod p on F_p* (index i ↔ residue i+1)",
            "A_isomorphic": "a ★ b = π^{-1}(π(a)*π(b) mod p); π uniform random permutation of 0..n-1; reject automorphisms (table equals genuine)",
            "B_output_perm": "y(a,b) = π(a*b mod p); inputs not permuted; reject identity π",
            "C_slot_relabel": "y(a,b) = πA(a)*πB(b) mod p; πA ≠ πB required",
            "D_random_comm": "shuffle upper-triangle and diagonal of genuine table; keep exact class frequencies; require commutative, balanced, nonassociative",
            "E_latin_nonassoc": "intercalate trades on the genuine Cayley table; require Latin, nonassociative, noncommutative, not a group",
            "E_latin_comm_nonassoc": "symmetric intercalate trades; require Latin, commutative, nonassociative, not a group",
            "F_assoc_damage": (
                "Select a preregistered fraction of unordered pairs {a≤b} of the genuine "
                f"table ({list(CORRUPT_FRACS)}). Shuffle values separately within diagonal "
                "cells and within off-diagonal pairs so class frequencies stay exact and "
                "the table stays symmetric. A singleton selection cannot change the table; "
                "then apply a 2-cycle of off-diagonal pairs (smallest frequency-preserving "
                "associativity damage; for n=10 at 1% this is 2/55 unordered pairs). "
                "Require associativity strictly damaged. Record actual n_modified."
            ),
            "G_noncyclic_group": (
                "Cayley table of C_a × C_b (additive). n=18 → C3×C6; n=28 → C2×C14; "
                "n=10 skipped (only cyclic abelian group). Require group and not cyclic."
            ),
        },
        "structure_matrix": STRUCTURE_MATRIX,
        "computational_priority": [
            "genuine baseline",
            "A isomorphic relabeling",
            "D random commutative balanced",
            "E nonassociative Latin square",
            "F associativity corruption series",
            "C independent-slot relabeling",
            "G noncyclic group",
            "B output permutation (readout invariance; last among listed because C3 predicts latent survival)",
        ],
        "omitted_conditions": [
            "p=31 (optional; declined before training)",
            "p=23",
            "fully i.i.d. non-commutative random tables beyond D/E/F",
            "Model D / architecture search",
        ],
        "construction_notes": {
            "F_equal_weight_shuffle": (
                "Diagonal and off-diagonal values are never mixed in the same shuffle, "
                "because they have multiplicity 1 vs 2 in the symmetric table. Mixing them "
                "was observed to destroy class balance and is not used."
            ),
            "F_singleton_fallback": (
                "round(0.01 * n(n+1)/2) equals 1 at n=10. Shuffling one stored value is a "
                "no-op. The locked fallback is a 2-cycle of off-diagonal pairs (2/55 "
                "unordered pairs). Actual differing-entry counts are stored per table."
            ),
            "E_random_trade_count": (
                "Number of intercalate trades is drawn uniformly from 1..min(8, n_candidates) "
                "so replication instances are not identical."
            ),
        },
        "stopping_rule": {
            "after_preregistered_controls_finish": "STOP and write the Stage-B report",
            "if_novel_exact_symbolic_family": "STOP EVEN EARLIER and report before continuing",
            "do_not": [
                "add more control types",
                "add more primes",
                "investigate p=23",
                "move to Collatz",
                "change architecture",
                "run Model D",
            ],
        },
        "planned_jobs": [{k: v for k, v in j.items() if k != "train_now"} | {"train_now": False} for j in jobs],
        "runtime_estimate": runtime,
        "training_status": "NOT STARTED — preregistration and table generation only",
        "run_dir": str(run_dir),
        "preregistration_written_before_training": True,
        "preregistration_written_before_inspecting_control_metrics": True,
    }


def _jsonable_generation(row: dict) -> dict:
    out = {k: v for k, v in row.items() if k != "table"}
    if "diagnostics" in out and out["diagnostics"] is not None:
        d = dict(out["diagnostics"])
        d.pop("class_counts", None)
        d["class_counts_preview"] = out["diagnostics"].get("class_counts", [])[:8]
        out["diagnostics"] = d
    return out


def generate_all_tables(run_dir: Path, jobs: list[dict]) -> dict:
    tables_dir = run_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    failures = []
    skipped = []
    for job in jobs:
        rec = generate_verified_table(job)
        if rec.get("skipped"):
            skipped.append(_jsonable_generation(rec))
            rows.append(_jsonable_generation(rec))
            continue
        if not rec.get("ok"):
            failures.append(_jsonable_generation(rec))
            rows.append(_jsonable_generation(rec))
            write_json(tables_dir / f"{job['job_id']}_FAILED.json", _jsonable_generation(rec))
            continue
        T = rec["table"]
        np.save(tables_dir / f"{job['job_id']}.npy", T)
        extras = rec.get("extras") or {}
        meta = _jsonable_generation(rec)
        write_json(tables_dir / f"{job['job_id']}.json", meta)
        if "pi" in extras:
            np.save(tables_dir / f"{job['job_id']}_pi.npy", np.array(extras["pi"], dtype=np.int64))
        if "pi_a" in extras:
            np.save(tables_dir / f"{job['job_id']}_pi_a.npy", np.array(extras["pi_a"], dtype=np.int64))
            np.save(tables_dir / f"{job['job_id']}_pi_b.npy", np.array(extras["pi_b"], dtype=np.int64))
        rows.append(meta)

    n_ok = sum(1 for r in rows if r.get("ok"))
    n_fail = len(failures)
    n_skip = len(skipped)
    coverage_ok = all(r.get("held_row", {}).get("all_classes_in_train", False) for r in rows if r.get("ok"))
    summary = {
        "n_planned": len(jobs),
        "n_ok": n_ok,
        "n_failed": n_fail,
        "n_skipped": n_skip,
        "all_ok_or_intentionally_skipped": n_fail == 0,
        "all_held_row_train_classes_complete": coverage_ok,
        "failures": failures,
        "skipped": skipped,
        "tables_dir": str(tables_dir),
        "rows": rows,
    }
    return summary


def print_structure_table() -> None:
    print()
    print("STRUCTURE: what each control preserves vs destroys")
    print("=" * 88)
    hdr = f"{'control':<26} {'kind':<42} {'preserves':<32} {'destroys'}"
    print(hdr)
    print("-" * 88)
    for row in STRUCTURE_MATRIX:
        pres = ", ".join(row["preserves"][:3]) + ("…" if len(row["preserves"]) > 3 else "")
        dest = ", ".join(row["destroys"][:2]) + ("…" if len(row["destroys"]) > 2 else "")
        print(f"{row['control']:<26} {row['kind']:<42} {pres:<32} {dest}")
    print()
    print("Full preserve/destroy lists are in control_predictions.json → structure_matrix.")


def print_generation_compact(summary: dict) -> None:
    print()
    print("GENERATED TABLES (structure only — no training)")
    print("=" * 110)
    print(
        f"{'job':<42} {'ok':<4} {'comm':<5} {'assoc':<6} {'A':<8} {'Latin':<6} "
        f"{'id':<4} {'group':<6} {'cyc':<4} {'trainCls':<8} {'seed'}"
    )
    print("-" * 110)
    for r in summary["rows"]:
        if r.get("skipped"):
            print(f"{r['job_id']:<42} skip {r.get('skip_reason', '')[:60]}")
            continue
        if not r.get("ok"):
            print(f"{r['job_id']:<42} FAIL construction")
            continue
        d = r["diagnostics"]
        h = r["held_row"]
        print(
            f"{r['job_id']:<42} {'Y':<4} {str(d['commutative'])[0]:<5} "
            f"{str(d['associative'])[0]:<6} {d['associativity_fraction']:<8.4f} "
            f"{str(d['latin'])[0]:<6} {str(d['identity_exists'])[0]:<4} "
            f"{str(d['is_group'])[0]:<6} {str(d['is_cyclic_group'])[0]:<4} "
            f"{h['n_train_classes']:<8} {r['seed']}"
        )


def sanity_genuine(p: int) -> dict:
    T = genuine_mul(p)
    from src.analysis.control_tables import diagnose, held_row_coverage

    d = diagnose(T)
    c = held_row_coverage(T)
    ok = d["is_cyclic_group"] and d["latin"] and c["all_classes_in_train"]
    return {"p": p, "ok": ok, "diagnostics": {k: d[k] for k in (
        "commutative", "associative", "associativity_fraction", "latin",
        "identity_exists", "inverses_exist", "is_group", "is_cyclic_group",
    )}, "held_row_all_classes": c["all_classes_in_train"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        action="store_true",
        help="FORBIDDEN in the preregistration step. Refused unless tables already exist and the user explicitly asked to train.",
    )
    args = parser.parse_args()
    if args.train:
        raise SystemExit(
            "Refusing --train. Stage B GPU training is locked until the "
            "preregistration is accepted. Re-run without --train."
        )

    jobs = planned_jobs()
    run_dir = new_run_dir(ROOT, "stage_b_controls")
    (run_dir / "tables").mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)

    preds = preregistration_payload(run_dir, jobs)
    pred_path = run_dir / "control_predictions.json"
    write_json(pred_path, preds)
    write_json(run_dir / "metrics" / "control_predictions.json", preds)
    write_yaml(run_dir / "config.yaml", {
        "stage": "B",
        "primes": list(PRIMES),
        "train": False,
        "master_seed": MASTER_SEED,
    })
    write_json(run_dir / "meta.json", {
        "stage": "B",
        "training_status": "not_started",
        "environment": environment_record(),
        "preregistration": str(pred_path),
    })

    print("PREREGISTRATION WRITTEN (before table inspection, before any training)")
    print(f"  {pred_path}")
    print(f"  primes: {list(PRIMES)}")
    print(f"  planned jobs: {len(jobs)}")
    print(f"  p=31 included: NO")
    print(f"  GPU training: NO")

    genuine_checks = [sanity_genuine(p) for p in PRIMES]
    write_json(run_dir / "metrics" / "genuine_sanity.json", genuine_checks)
    if not all(g["ok"] for g in genuine_checks):
        raise SystemExit(f"genuine mul diagnostics failed: {genuine_checks}")

    summary = generate_all_tables(run_dir, jobs)
    write_json(run_dir / "metrics" / "control_generation.json", {
        k: v for k, v in summary.items() if k != "rows"
    } | {"n_rows": len(summary["rows"])})
    write_json(run_dir / "metrics" / "control_generation_rows.json", summary["rows"])

    print_structure_table()
    print_generation_compact(summary)
    print()
    print("HELD-ROW: all generated OK tables have every output class in train:",
          summary["all_held_row_train_classes_complete"])
    print("Construction failures:", summary["n_failed"])
    print("Intentional skips (G at p=11):", summary["n_skipped"])
    rt = preds["runtime_estimate"]
    print(
        f"Expected GPU runtime AFTER approval: "
        f"{rt['est_minutes_if_256_scales']}–{rt['est_minutes_if_launch_bound']} min sequential "
        f"({rt['n_training_jobs']} jobs, {rt['n_1024_equivalents']}×1024-eq)."
    )
    print()
    print("STOP. No GPU training. No new primes. No Collatz. No p=23.")
    print(f"Preregistration path:\n  {pred_path}")


if __name__ == "__main__":
    _fix_path()
    main()
