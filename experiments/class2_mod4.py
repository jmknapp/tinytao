#!/usr/bin/env python3
"""Prospective H_CLASS2_MOD4 test: six unseen primes, Stage A protocol.

Writes predictions.json BEFORE any new training. Does not scramble, retrain
old populations, change architecture, or move to Collatz / Model D / p=7.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from experiments.stage_a_primes import held_row_coverage, run_prime
from src.analysis.class2_mod4 import (
    CLASS2_CRITERIA,
    EPOCHS,
    H_CLASS2_MOD4,
    OLD_PRIMES,
    POPULATION,
    PREDICTIONS,
    PROSPECTIVE_PRIMES,
    category_from_k,
    certify_class2_net,
    prime_card,
)
from src.analysis.fourier import discrete_log_table
from src.analysis.group_law import fit_dlog_circle_batch, load_population, units_mod
from src.analysis.level4_h1 import effective_winding
from src.analysis.radial_quotient import winding_scores
from src.analysis.stats import wilson_interval
from src.config import ExperimentConfig
from src.hardware import resolve_device
from src.repro import environment_record
from src.storage import new_run_dir, write_json
from src.symbolic_f13 import classify_pairs, from_fit
from src.visualization.class2_mod4 import (
    plot_class2_vs_mod4,
    plot_embedding_plane,
    plot_parity_consistency,
    plot_rate_vs_p,
    plot_shell_diagram,
    plot_winding_gcd,
)

STAGE_A_RUN = ROOT / "results" / "runs" / "20260909_112501_stage_a_primes"
P13_HELD_ROW = ROOT / "results" / "runs" / "20260909_104844_held_row"
P17_SECONDS = 27.268501953125
P17_DOMAIN = 256.0


def old_ckpt(p: int) -> Path:
    if p == 13:
        return P13_HELD_ROW / "complex_h1_held_row" / "checkpoints" / "successful.pt"
    return STAGE_A_RUN / f"complex_h1_p{p}_held_row" / "checkpoints" / "successful.pt"


def old_train_counts() -> dict[int, dict]:
    """Locked Stage A population counts. p=13 exact includes unsaved nets."""
    return {
        7: {"n_exact": 554, "n_saved": 554, "seconds": 30.877},
        11: {"n_exact": 553, "n_saved": 553, "seconds": 29.982},
        13: {"n_exact": 375, "n_saved": 256, "seconds": 31.107, "note": "finalize cap 256/375"},
        17: {"n_exact": 516, "n_saved": 516, "seconds": 27.269},
    }


def census_locked() -> dict:
    from src.analysis.class2_mod4 import CENSUS

    return dict(CENSUS)


def prerun_document() -> dict:
    cards = [prime_card(p) for p in PROSPECTIVE_PRIMES]
    old_cards = [prime_card(p) for p in OLD_PRIMES]
    coverage = [held_row_coverage(c["p"], c["held_residue"]) for c in cards]
    runtime = []
    total_scaled = 0.0
    for c in cards:
        scale = c["n_domain"] / P17_DOMAIN
        est = round(P17_SECONDS * scale, 1)
        total_scaled += est
        runtime.append(
            {
                "p": c["p"],
                "est_seconds_scaled_from_p17": est,
                "domain_vs_p17": round(scale, 3),
                "params": c["params_h1"],
            }
        )
    g_rows = []
    for c in cards + old_cards:
        v = c["primitive_root_verification"]
        g_rows.append(
            {
                "p": c["p"],
                "g": v["g"],
                "order": v["order"],
                "order_equals_p_minus_1": v["order_equals_p_minus_1"],
            }
        )
    all_g_ok = all(r["order_equals_p_minus_1"] for r in g_rows)
    coverage_ok = all(
        cov["all_classes_in_train"]
        and cov["all_classes_in_test"]
        and cov["a0_appears_as_b_in_train"]
        and cov["a0_never_as_a_in_train"]
        and cov["train_has_all_residues_as_a_or_b"]
        for cov in coverage
    )
    return {
        "hypothesis": H_CLASS2_MOD4,
        "predictions": PREDICTIONS,
        "class2_criteria": CLASS2_CRITERIA,
        "prime_list_locked": list(PROSPECTIVE_PRIMES),
        "prime_decision": (
            "Full six-prime test. Smaller four-prime subset was not selected. "
            "Decision made before training."
        ),
        "old_primes_not_prospective": list(OLD_PRIMES),
        "population": POPULATION,
        "epochs": EPOCHS,
        "optimizer": "adam",
        "lr": 0.003,
        "split": "held_row a0=p-1",
        "architecture": "Model C H=1, Xavier, no dlog/Legendre init",
        "params_formula": "5(p-1)",
        "latent_dim": 2,
        "not_in_this_run": [
            "multiplication-table scramble",
            "p=7 mechanistic follow-up",
            "extra primes beyond the locked six",
            "Model D",
            "Collatz",
            "architecture changes",
        ],
        "prime_cards": cards,
        "old_prime_cards": old_cards,
        "held_row_coverage": coverage,
        "coverage_all_ok": coverage_ok,
        "primitive_roots": g_rows,
        "all_primitive_roots_verified": all_g_ok,
        "runtime_estimate": {
            "baseline": "Stage A p=17 H=1 held-row, 1024 nets, 8000 steps, 27.3s",
            "scaled_from_domain": runtime,
            "sum_scaled_seconds": round(total_scaled, 1),
            "stage_a_was_launch_bound_about_30s_each": True,
            "wall_clock_guess_minutes": [3, 12],
            "population": POPULATION,
            "epochs": EPOCHS,
        },
        "census_thresholds_locked": census_locked(),
    }


def classify_checkpoint(ckpt_path: Path, p: int, figs: Path | None, tag: str) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = load_population(ckpt, torch.device("cpu"))
    n = p - 1
    dlog = discrete_log_table(p)
    E = model.E.detach().cpu()[:, :, 0, :].numpy().astype(np.float64)
    n_saved = int(E.shape[0])
    batch = fit_dlog_circle_batch(E, dlog, n)
    k_eff = effective_winding(batch["k"], batch["conjugate"], n)
    scores = winding_scores(E, dlog, n)
    units = set(units_mod(n))
    cats = []
    certs = []
    gcd_hist: dict[int, int] = {}
    n_faithful_sym = 0
    rng = np.random.default_rng(p)
    a = np.arange(1, p).repeat(n)
    b = np.tile(np.arange(1, p), n)
    y = ((a * b) % p) - 1
    plotted = False
    for i in range(n_saved):
        ki = int(k_eff[i])
        gcd = math.gcd(ki, n) if ki else n
        gcd_hist[gcd] = gcd_hist.get(gcd, 0) + 1
        order = np.argsort(scores["r2_k"][i])[::-1]
        k2 = int(order[1])
        cat = category_from_k(ki, n, float(batch["r2"][i]), float(scores["r2_k"][i, k2]), k2 in units)
        cats.append(cat)
        if gcd == 1:
            fit = {
                "k": int(batch["k"][i]),
                "phi": float(batch["phi"][i]),
                "scale": float(batch["scale"][i]),
                "conjugate": bool(batch["conjugate"][i]),
            }
            pred = classify_pairs(from_fit(p, fit), a, b, dlog)
            if bool((pred == y).all()):
                n_faithful_sym += 1
        if gcd == 2:
            cert = certify_class2_net(E[i], ki, float(batch["phi"][i]), p, rng)
            cert["net"] = i
            certs.append(cert)
            if cert["certified"] and figs is not None and not plotted:
                plot_embedding_plane(
                    E[i],
                    p,
                    ki,
                    figs / f"class2_embed_p{p}.png",
                    f"Class-2 embedding p={p} k={ki} ({tag})",
                )
                plot_shell_diagram(
                    E[i],
                    p,
                    figs / f"class2_shells_p{p}.png",
                    f"Product phase vs radius p={p} ({tag})",
                )
                plotted = True
    n_c2 = int(sum(1 for c in certs if c["certified"]))
    n_other_nonunit = int(sum(1 for c in cats if c == "C_other_nonunit"))
    n_amb = int(sum(1 for c in cats if c == "D_ambiguous"))
    breaks: dict[str, int] = {}
    for c in certs:
        if not c["certified"]:
            bkey = str(c["break_at"])
            breaks[bkey] = breaks.get(bkey, 0) + 1
    parity_vals = [c["radial_parity_consistency"] for c in certs]
    quot_err = [1.0 - c["quotient_phase_acc"] for c in certs]
    return {
        "p": p,
        "n_saved": n_saved,
        "n_faithful_symbolic": n_faithful_sym,
        "n_gcd2": int(len(certs)),
        "n_class2_certified": n_c2,
        "n_other_nonunit_gcd_gt2": n_other_nonunit,
        "n_ambiguous": n_amb,
        "gcd_hist": {str(k): v for k, v in sorted(gcd_hist.items())},
        "gcd2_break_counts": breaks,
        "radial_parity_consistency_gcd2": float(np.mean(parity_vals)) if parity_vals else None,
        "median_class2_quotient_error": float(np.median(quot_err)) if quot_err else None,
        "median_certified_quotient_error": (
            float(np.median([1.0 - c["quotient_phase_acc"] for c in certs if c["certified"]]))
            if n_c2
            else None
        ),
        "n_decoder_exact_among_gcd2": int(sum(1 for c in certs if c["decoder_exact"])),
        "mean_shell_xor_acc_gcd2": float(np.mean([c["shell_xor_acc"] for c in certs])) if certs else None,
        "mean_phase_only_acc_gcd2": float(np.mean([c["phase_only_even_lift_acc"] for c in certs])) if certs else None,
        "mean_unit_norm_decoder_acc_gcd2": float(np.mean([c["unit_norm_decoder_acc"] for c in certs])) if certs else None,
        "mean_destroying_decoder_acc_gcd2": (
            float(np.mean([c["parity_destroying_decoder_acc"] for c in certs])) if certs else None
        ),
        "category_counts": {
            "A_faithful_candidate": int(sum(1 for c in cats if c == "A_faithful_candidate")),
            "B_class2_candidate": int(sum(1 for c in cats if c == "B_class2_candidate")),
            "C_other_nonunit": n_other_nonunit,
            "D_ambiguous": n_amb,
        },
        "certs_gcd2": certs,
        "latent_dim": 2,
        "params": 5 * n,
    }


def summary_row(p: int, n_exact: int, n_pop: int, classif: dict, source_kind: str, extra: dict | None = None) -> dict:
    card = prime_card(p)
    pe, lo_e, hi_e = wilson_interval(n_exact, n_pop)
    pf, lo_f, hi_f = wilson_interval(classif["n_faithful_symbolic"], n_pop)
    pc, lo_c, hi_c = wilson_interval(classif["n_class2_certified"], n_pop)
    n_other = n_exact - classif["n_faithful_symbolic"] - classif["n_class2_certified"]
    if classif["n_saved"] < n_exact:
        n_other = max(0, classif["n_saved"] - classif["n_faithful_symbolic"] - classif["n_class2_certified"])
        n_other = n_exact - classif["n_faithful_symbolic"] - classif["n_class2_certified"]
    row = {
        "p": p,
        "p_mod_4": card["p_mod_4"],
        "factorization": card["p_minus_1_factorization_pretty"],
        "phi_n": card["phi_n"],
        "n_gcd2_windings_available": card["n_gcd2_k"],
        "C_n_iso_C_m_x_C2": card["C_n_iso_C_m_x_C2"],
        "n_exact": n_exact,
        "n_faithful_level4": classif["n_faithful_symbolic"],
        "n_class2": classif["n_class2_certified"],
        "n_other_exact": n_other,
        "p_exact": pe,
        "p_exact_lo": lo_e,
        "p_exact_hi": hi_e,
        "p_faithful": pf,
        "p_faithful_lo": lo_f,
        "p_faithful_hi": hi_f,
        "p_class2": pc,
        "p_class2_lo": lo_c,
        "p_class2_hi": hi_c,
        "median_faithful_hom_error_deg": (extra or {}).get("median_faithful_hom_error_deg"),
        "median_class2_quotient_error": classif["median_certified_quotient_error"],
        "radial_parity_consistency_gcd2": classif["radial_parity_consistency_gcd2"],
        "symbolic_class2_replacement_rate": (
            classif["n_class2_certified"] / classif["n_gcd2"] if classif["n_gcd2"] else None
        ),
        "gcd_hist": classif["gcd_hist"],
        "gcd2_break_counts": classif["gcd2_break_counts"],
        "source_kind": source_kind,
        "n_saved": classif["n_saved"],
        "params": classif["params"],
        "g": card["primitive_root"],
        "n": card["n"],
        "m": card["m"],
        "n_gcd2_candidates": classif["n_gcd2"],
    }
    if extra:
        row.update({k: v for k, v in extra.items() if k not in row})
    return row


def outcome_letter(prospective_rows: list[dict]) -> str:
    pos = [r for r in prospective_rows if r["p_mod_4"] == 3]
    neg = [r for r in prospective_rows if r["p_mod_4"] == 1]
    any_pos = any(r["n_class2"] > 0 for r in pos)
    any_neg = any(r["n_class2"] > 0 for r in neg)
    gcd2_uncert = any((r.get("n_gcd2_candidates") or 0) > 0 and r["n_class2"] == 0 for r in pos)
    other_family = any((r.get("n_other_nonunit") or 0) > 0 for r in prospective_rows)
    if other_family and not any_pos and not any_neg:
        return "E"
    if any_pos and not any_neg:
        return "A"
    if any_neg:
        return "B"
    if gcd2_uncert:
        return "C"
    if not any_pos:
        return "D"
    return "A"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "class2_mod4.yaml")
    parser.add_argument("--finish-run", type=Path, default=None)
    parser.add_argument("--prereg-only", action="store_true")
    args = parser.parse_args()
    cfg = ExperimentConfig.from_yaml(args.config)
    cfg.population.size = POPULATION
    cfg.train.epochs = EPOCHS

    if args.finish_run is None:
        run_dir = new_run_dir(ROOT, cfg.run_name)
        prerun = prerun_document()
        prerun["environment"] = environment_record(str(ROOT))
        prerun["predictions_path"] = str(run_dir / "metrics" / "predictions.json")
        write_json(run_dir / "metrics" / "predictions.json", prerun)
        write_json(run_dir / "meta.json", {**environment_record(str(ROOT)), "hypothesis": "H_CLASS2_MOD4", "locked": True})
        print(f"run directory: {run_dir}")
        print(f"preregistration: {run_dir / 'metrics' / 'predictions.json'}")
        print("\n===== H_CLASS2_MOD4 locked =====")
        print(H_CLASS2_MOD4)
        print("\n===== P1–P10 =====")
        for k, v in PREDICTIONS.items():
            print(f"  {k}. {v}")
        print("\n===== Class-2 certificate =====")
        for k, v in CLASS2_CRITERIA.items():
            print(f"  {k}: {v}")
        print("\n===== Locked prime list =====")
        print("  prospective:", PROSPECTIVE_PRIMES)
        print("  decision: full six-prime test (not the four-prime fallback)")
        print("  population", POPULATION, "epochs", EPOCHS)
        print("\n===== Pre-run arithmetic =====")
        print(
            f"{'p':>3} {'mod4':>4} {'n':>3} {'m':>3} {'n fact':<12} {'phi':>4} "
            f"{'units':>5} {'gcd2':>4} {'Cn≅Cm×C2':>9} {'train':>5} {'held':>4} {'params':>6} {'g':>3} {'ord=n':>5}"
        )
        for c in prerun["prime_cards"]:
            v = c["primitive_root_verification"]
            print(
                f"{c['p']:3d} {c['p_mod_4']:4d} {c['n']:3d} {c['m']:3d} "
                f"{c['p_minus_1_factorization_pretty']:<12} {c['phi_n']:4d} "
                f"{c['n_unit_k']:5d} {c['n_gcd2_k']:4d} {str(c['C_n_iso_C_m_x_C2']):>9} "
                f"{c['train_pair_count']:5d} {c['held_row_pair_count']:4d} {c['params_h1']:6d} "
                f"{c['primitive_root']:3d} {str(v['order_equals_p_minus_1']):>5}"
            )
        print("\nPrimitive roots (prospective + old):")
        for r in prerun["primitive_roots"]:
            print(f"  p={r['p']} g={r['g']} order={r['order']} verified={r['order_equals_p_minus_1']}")
        print("all primitive roots verified:", prerun["all_primitive_roots_verified"])
        print("held-row coverage all ok:", prerun["coverage_all_ok"])
        for cov in prerun["held_row_coverage"]:
            print(
                f"  p={cov['p']}: a0_never_a={cov['a0_never_as_a_in_train']} "
                f"a0_as_b={cov['a0_appears_as_b_in_train']} "
                f"classes_train={cov['all_classes_in_train']} classes_test={cov['all_classes_in_test']}"
            )
        est = prerun["runtime_estimate"]
        print(
            f"\nRuntime estimate: scaled-from-p17 sum {est['sum_scaled_seconds']}s "
            f"(~{est['sum_scaled_seconds']/60:.1f} min); Stage A was ~30s/prime launch-bound. "
            f"Guess {est['wall_clock_guess_minutes'][0]}–{est['wall_clock_guess_minutes'][1]} min."
        )
        print(f"Path to preregistration: {run_dir / 'metrics' / 'predictions.json'}")
        if args.prereg_only:
            print("prereg-only: stopping before training.")
            return
        if not prerun["all_primitive_roots_verified"] or not prerun["coverage_all_ok"]:
            raise RuntimeError("preregistration checks failed; not training")
        print("\n===== Training starts now (protocol frozen) =====")
    else:
        run_dir = args.finish_run
        print(f"re-using run dir {run_dir}")

    device = resolve_device(cfg.device)
    by_p = {}
    if args.finish_run is None:
        for p in PROSPECTIVE_PRIMES:
            by_p[p] = run_prime(cfg, p, device, run_dir)
            write_json(
                run_dir / "metrics" / "train_progress.json",
                {str(k): v["train"] for k, v in by_p.items()},
            )
    else:
        for p in PROSPECTIVE_PRIMES:
            path = run_dir / f"complex_h1_p{p}_held_row" / "metrics" / "stage_a_prime.json"
            with open(path, encoding="utf-8") as f:
                by_p[p] = json.load(f)

    figs = run_dir / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    rows = []
    classifs = {}
    print("\n===== Classification and Class-2 certificates =====")
    for p in PROSPECTIVE_PRIMES:
        ckpt = run_dir / f"complex_h1_p{p}_held_row" / "checkpoints" / "successful.pt"
        n_exact = int(by_p[p]["train"]["n_exact"])
        if n_exact == 0 or not ckpt.exists():
            classif = {
                "p": p,
                "n_saved": 0,
                "n_faithful_symbolic": 0,
                "n_gcd2": 0,
                "n_class2_certified": 0,
                "n_other_nonunit_gcd_gt2": 0,
                "n_ambiguous": 0,
                "gcd_hist": {},
                "gcd2_break_counts": {},
                "radial_parity_consistency_gcd2": None,
                "median_class2_quotient_error": None,
                "median_certified_quotient_error": None,
                "certs_gcd2": [],
                "params": 5 * (p - 1),
                "category_counts": {
                    "A_faithful_candidate": 0,
                    "B_class2_candidate": 0,
                    "C_other_nonunit": 0,
                    "D_ambiguous": 0,
                },
            }
        else:
            classif = classify_checkpoint(ckpt, p, figs, "prospective")
        classifs[p] = {k: v for k, v in classif.items() if k != "certs_gcd2"}
        write_json(run_dir / "metrics" / f"class2_p{p}.json", classif)
        unit = (by_p[p].get("exact") or {}).get("unit_exact") or {}
        extra = {
            "median_faithful_hom_error_deg": unit.get("mean_abs_centered_deg"),
            "seconds": by_p[p]["train"].get("seconds"),
            "n_gcd2_candidates": classif["n_gcd2"],
        }
        row = summary_row(p, n_exact, POPULATION, classif, "prospective", extra)
        rows.append(row)
        print(
            f"p={p} ≡{p % 4} (mod 4)  exact {n_exact}/1024  faithful {classif['n_faithful_symbolic']}  "
            f"gcd2 {classif['n_gcd2']}  Class-2 {classif['n_class2_certified']}  "
            f"breaks {classif['gcd2_break_counts']}"
        )

    print("\n===== OLD DATA (same certificate, not prospective) =====")
    old_counts = old_train_counts()
    old_classifs = {}
    for p in OLD_PRIMES:
        classif = classify_checkpoint(old_ckpt(p), p, figs, "old")
        old_classifs[p] = {k: v for k, v in classif.items() if k != "certs_gcd2"}
        write_json(run_dir / "metrics" / f"class2_old_p{p}.json", classif)
        oc = old_counts[p]
        extra = {"n_gcd2_candidates": classif["n_gcd2"], "old_note": oc.get("note")}
        row = summary_row(p, oc["n_exact"], POPULATION, classif, "old", extra)
        rows.append(row)
        print(
            f"OLD p={p} ≡{p % 4} (mod 4)  exact {oc['n_exact']}/1024 saved {classif['n_saved']}  "
            f"faithful {classif['n_faithful_symbolic']}  gcd2 {classif['n_gcd2']}  "
            f"Class-2 {classif['n_class2_certified']}  breaks {classif['gcd2_break_counts']}"
        )

    prospective_rows = [r for r in rows if r["source_kind"] == "prospective"]
    letter = outcome_letter(prospective_rows)
    payload = {
        "hypothesis": H_CLASS2_MOD4,
        "outcome_letter": letter,
        "outcome_legend": {
            "A": "Class-2 on p≡3 mod 4 only",
            "B": "Class-2 in both congruence classes",
            "C": "gcd=2 exact at p≡3 mod 4 but certificate fails",
            "D": "no new Class-2 despite algebraic availability",
            "E": "different exact family (inspect C/D categories)",
        },
        "rows": rows,
        "prospective_classifications": classifs,
        "old_classifications": old_classifs,
        "stop": "Do not add primes, scramble, follow up p=7, or move to Collatz.",
    }
    write_json(run_dir / "metrics" / "class2_mod4.json", payload)

    plot_rate_vs_p(rows, "p_exact", "p_exact_lo", "p_exact_hi", "P(domain exact)", "P(domain exact) vs p", figs / "p_exact_vs_p.png")
    plot_rate_vs_p(rows, "p_faithful", "p_faithful_lo", "p_faithful_hi", "P(Level-4 faithful)", "P(faithful) vs p", figs / "p_faithful_vs_p.png")
    plot_rate_vs_p(rows, "p_class2", "p_class2_lo", "p_class2_hi", "P(Class-2)", "P(Class-2) vs p", figs / "p_class2_vs_p.png")
    plot_winding_gcd(rows, figs / "winding_gcd.png")
    plot_parity_consistency(rows, figs / "radius_parity.png")
    plot_class2_vs_mod4(rows, figs / "class2_vs_mod4.png")

    print("\n===== REQUIRED SUMMARY TABLE =====")
    print(
        "src  p  mod4  fact(p-1)  phi  #gcd2k  exact  faithful  Class-2  other  "
        "P(exact)  P(faithful)  P(Class-2)  radParity  C2replace"
    )
    for r in rows:
        src = "OLD" if r["source_kind"] == "old" else "NEW"
        print(
            f"{src:3} {r['p']:3d} {r['p_mod_4']:4d} {r['factorization']:<12} {r['phi_n']:3d} "
            f"{r['n_gcd2_windings_available']:6d} {r['n_exact']:5d} {r['n_faithful_level4']:8d} "
            f"{r['n_class2']:7d} {r['n_other_exact']:5d} {r['p_exact']:.3f} {r['p_faithful']:.3f} "
            f"{r['p_class2']:.3f} {r['radial_parity_consistency_gcd2']} "
            f"{r['symbolic_class2_replacement_rate']}"
        )
    print(f"\nOUTCOME {letter}. STOP.")
    print("wrote", run_dir / "metrics" / "class2_mod4.json")


if __name__ == "__main__":
    main()
