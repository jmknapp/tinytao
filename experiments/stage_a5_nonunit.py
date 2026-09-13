#!/usr/bin/env python3
"""Stage A.5 Step 1: census and geometry of non-unit exact H=1 nets.

Uses saved p=7 and p=11 exact checkpoints only. No retraining.
Hypotheses are written to disk before embeddings are loaded.
Does not run ablations, symbolic extraction, or scrambled controls.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.analysis.fourier import discrete_log_table, primitive_root
from src.analysis.group_law import fit_dlog_circle_batch, load_population, units_mod
from src.analysis.level4_h1 import effective_winding
from src.analysis.radial_quotient import (
    analyze_one_net,
    fiber_members,
    kernel_size,
    n_phases,
    population_descriptor_consistency,
    winding_scores,
)
from src.analysis.stage_a5_hypotheses import (
    CENSUS_THRESHOLDS,
    HYPOTHESES,
    STAGE_A5_PRIMES,
    STAGE_A_RUN,
)
from src.hardware import resolve_device
from src.repro import environment_record
from src.storage import new_run_dir, write_json
from src.tasks.modular import ModularMultiplicationStar
from src.visualization.stage_a5 import (
    plot_aligned_overlay,
    plot_census_bars,
    plot_decision_regions,
    plot_embedding_plane,
    plot_metric_hist,
    plot_phase_vs_ideal,
    plot_polar_r_theta,
    plot_product_plane,
    plot_radius_by_fiber,
    plot_radius_vs_dlog,
)

STAGE_A_DIR = ROOT / STAGE_A_RUN


def theoretical_kernel_card(p: int) -> dict:
    n = p - 1
    g = int(primitive_root(p))
    dlog = discrete_log_table(p)
    units = units_mod(n)
    non_units = [k for k in range(n) if k not in units]
    by_k = {}
    for k in non_units:
        d = kernel_size(k, n)
        fibers = fiber_members(k, p)
        by_k[str(k)] = {
            "k": k,
            "gcd": d,
            "kernel_size": d,
            "n_phases": n_phases(k, n),
            "fibers": {str(f): members for f, members in fibers.items()},
            "fibers_are_a_minus_a": all(
                len(m) == 2 and (m[0] + m[1]) % p == 0 for m in fibers.values()
            )
            if d == 2
            else False,
        }
    return {
        "p": p,
        "n": n,
        "g": g,
        "dlog": {str(a): int(dlog[a]) for a in range(1, p)},
        "units": units,
        "non_units": non_units,
        "by_k": by_k,
    }


def locked_payload() -> dict:
    return {
        "step": 1,
        "hypotheses": HYPOTHESES,
        "census_thresholds": CENSUS_THRESHOLDS,
        "theoretical_kernels": {str(p): theoretical_kernel_card(p) for p in STAGE_A5_PRIMES},
        "not_in_this_step": [
            "radial/phase ablations",
            "within-fiber radius swaps",
            "symbolic radial replacement",
            "scrambled controls",
            "Stage B",
        ],
        "source_run": str(STAGE_A_DIR),
    }


def strip_heavy(row: dict) -> dict:
    skip = {"xy", "W_xy", "b", "embed_radius", "embed_theta"}
    return {k: v for k, v in row.items() if k not in skip}


def summarize_kind(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    def mean(key):
        return float(np.mean([r[key] for r in rows]))

    k_counts = dict(Counter(r["k_eff"] for r in rows))
    return {
        "n": len(rows),
        "k_counts": {str(k): int(c) for k, c in sorted(k_counts.items())},
        "mean_r2": mean("r2"),
        "mean_angular_residual_deg": mean("angular_residual_deg"),
        "mean_radius_cv": mean("radius_cv"),
        "mean_hom_deg": mean("mean_abs_centered_deg"),
        "mean_hom_train_deg": mean("mean_abs_centered_train_deg"),
        "mean_hom_held_deg": mean("mean_abs_centered_held_deg"),
        "frac_symbolic_roots_exact": float(np.mean([r["symbolic_roots_exact"] for r in rows])),
        "mean_quotient_acc": mean("quotient_acc_from_product_phase"),
        "mean_phase_only_acc": mean("target_acc_phase_only"),
        "mean_radius_only_acc": mean("target_acc_radius_only"),
        "mean_joint_acc": mean("target_acc_phase_and_radius"),
        "mean_H_target_given_phase": mean("H_target_given_phase"),
        "mean_H_target_given_phase_radius": mean("H_target_given_phase_radiusbin"),
        "mean_H_target": mean("H_target"),
        "frac_joint_exact": float(np.mean([r["target_acc_phase_and_radius"] >= 1.0 - 1e-12 for r in rows])),
        "frac_quotient_exact": float(np.mean([r["quotient_acc_from_product_phase"] >= 1.0 - 1e-12 for r in rows])),
        "mean_unit_h_acc": mean("unit_normalize_h_frozen_W_acc"),
        "mean_meanr_h_acc": mean("mean_radius_h_frozen_W_acc"),
        "mean_W_angle_spread_deg": mean("mean_W_angle_spread_within_fiber_deg"),
        "mean_W_radius_ratio": mean("mean_W_radius_ratio_within_fiber"),
        "mean_within_fiber_sep": mean("mean_within_fiber_sep"),
        "frac_fibers_separated": mean("frac_fibers_radius_ratio_gt_1_2"),
        "mean_mixed_parity_acc": mean("mixed_vs_target_dlog_parity_acc"),
        "frac_phase_bins_radius_disjoint": mean("frac_phase_bins_radius_disjoint"),
        "frac_domain_acc_check": mean("net_acc_check"),
    }


def analyze_prime(p: int, device, figs: Path) -> dict:
    ckpt_path = STAGE_A_DIR / f"complex_h1_p{p}_held_row" / "checkpoints" / "successful.pt"
    with open(STAGE_A_DIR / f"complex_h1_p{p}_held_row" / "metrics" / "stage_a_prime.json") as f:
        prev = json.load(f)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = load_population(ckpt, torch.device("cpu"))
    n = p - 1
    dlog = discrete_log_table(p)
    E = model.E.detach().cpu()[:, :, 0, :].numpy().astype(np.float64)
    W = model.W_out.detach().cpu().numpy().astype(np.float64)
    B = model.b_out.detach().cpu().numpy().astype(np.float64)
    batch = fit_dlog_circle_batch(E, dlog, n)
    k_eff = effective_winding(batch["k"], batch["conjugate"], n)
    scores = winding_scores(E, dlog, n)
    n_pop = E.shape[0]
    rows = []
    for i in range(n_pop):
        rows.append(
            analyze_one_net(
                E[i],
                W[i],
                B[i],
                p,
                int(k_eff[i]),
                float(batch["phi"][i]),
                float(batch["r2"][i]),
                bool(batch["conjugate"][i]),
                float(batch["scale"][i]),
                scores["r2_k"][i],
                p - 1,
                CENSUS_THRESHOLDS,
            )
        )
    kinds = Counter(r["kind"] for r in rows)
    by_kind = {k: [r for r in rows if r["kind"] == k] for k in ("faithful_unit", "non_unit", "ambiguous")}
    nonunit_by_k = defaultdict(list)
    for r in by_kind["non_unit"]:
        nonunit_by_k[r["k_eff"]].append(r)

    desc = {}
    for k, group in nonunit_by_k.items():
        desc[str(k)] = population_descriptor_consistency(group, p, k)

    # Population: after per-net polarity, all fibers agree on dlog parity of large radius
    dlog_t = discrete_log_table(p)
    fiber_agree = []
    for r in by_kind["non_unit"]:
        bits = [int(dlog_t[fr["large_radius_residue"]]) % 2 for fr in r["fibers"]]
        fiber_agree.append(len(set(bits)) == 1)
    polarity_consistency = {
        "n_non_unit": len(by_kind["non_unit"]),
        "frac_nets_all_fibers_same_dlog_parity_for_large_radius": float(np.mean(fiber_agree)) if fiber_agree else float("nan"),
    }

    print(f"\n=== p={p} census n_saved_exact={n_pop} (population exact {prev['train']['n_exact']}/1024) ===")
    print("  kinds", dict(kinds))
    print("  non-unit k", {k: len(v) for k, v in nonunit_by_k.items()})
    if by_kind["non_unit"]:
        s = summarize_kind(by_kind["non_unit"])
        print(
            f"  non-unit: R²={s['mean_r2']:.3f} hom={s['mean_hom_deg']:.2f}° "
            f"quot={s['mean_quotient_acc']:.3f} joint={s['mean_joint_acc']:.3f} "
            f"H|phase={s['mean_H_target_given_phase']:.3f} H|phase,r={s['mean_H_target_given_phase_radius']:.3f} "
            f"unit_h={s['mean_unit_h_acc']:.3f} meanr_h={s['mean_meanr_h_acc']:.3f} "
            f"fiber_agree={polarity_consistency['frac_nets_all_fibers_same_dlog_parity_for_large_radius']:.3f}"
        )

    pfigs = figs / f"p{p}"
    plot_census_bars({k: kinds[k] for k in ("faithful_unit", "non_unit", "ambiguous") if k in kinds}, pfigs / "census.png", f"p={p} exact H=1 winding census")
    if by_kind["non_unit"]:
        plot_metric_hist(np.array([r["quotient_acc_from_product_phase"] for r in by_kind["non_unit"]]), pfigs / "hist_quotient_acc.png", "product-phase quotient accuracy", f"p={p} non-unit: does product phase recover the quotient class?")
        plot_metric_hist(np.array([r["target_acc_phase_and_radius"] for r in by_kind["non_unit"]]), pfigs / "hist_joint_acc.png", "phase+radius target accuracy", f"p={p} non-unit: joint discrete decode")
        plot_metric_hist(np.array([r["H_target_given_phase"] for r in by_kind["non_unit"]]), pfigs / "hist_H_phase.png", "H(target | product phase class) bits", f"p={p} non-unit conditional entropy given phase")
        plot_metric_hist(np.array([r["mean_within_fiber_sep"] for r in by_kind["non_unit"]]), pfigs / "hist_fiber_sep.png", "mean within-fiber (rmax-rmin)/rmean", f"p={p} non-unit radius separation inside fibers")
        plot_metric_hist(np.array([r["unit_normalize_h_frozen_W_acc"] for r in by_kind["non_unit"]]), pfigs / "hist_unit_h.png", "accuracy after unit-normalizing h, frozen W,b", f"p={p} non-unit: angle-only products into learned readout")
        plot_metric_hist(np.array([r["mean_radius_h_frozen_W_acc"] for r in by_kind["non_unit"]]), pfigs / "hist_meanr_h.png", "accuracy after setting |h| to mean, frozen W,b", f"p={p} non-unit: equal-radius products into learned readout")

    # Representatives: one per non-unit k, plus one faithful
    reps = []
    for k, group in sorted(nonunit_by_k.items()):
        group_sorted = sorted(group, key=lambda r: -r["mean_within_fiber_sep"])
        reps.append(("non_unit", k, group_sorted[0], True))
        if len(group_sorted) > 1:
            reps.append(("non_unit", k, group_sorted[len(group_sorted) // 2], False))
    if by_kind["faithful_unit"]:
        reps.append(("faithful_unit", by_kind["faithful_unit"][0]["k_eff"], by_kind["faithful_unit"][0], True))

    for kind, k, row, primary in reps:
        xy = np.array(row["xy"], dtype=np.float64)
        tag = f"{kind}_k{k}" + ("_sepmax" if primary and kind == "non_unit" else "")
        plot_embedding_plane(xy, p, k, pfigs / f"{tag}_xy.png", f"p={p} {kind} k={k}  z(a) in C")
        plot_polar_r_theta(xy, p, k, pfigs / f"{tag}_polar.png", f"p={p} {kind} k={k}  θ vs radius")
        plot_radius_vs_dlog(xy, p, k, pfigs / f"{tag}_r_dlog.png", f"p={p} {kind} k={k}  radius vs dlog")
        plot_radius_by_fiber(xy, p, k, pfigs / f"{tag}_r_fiber.png", f"p={p} {kind} k={k}  radius by phase fiber")
        plot_phase_vs_ideal(xy, p, k, row["phi"], pfigs / f"{tag}_phase.png", f"p={p} {kind} k={k}  learned vs ideal phase")
        if primary:
            Wrep = np.array(row["W_xy"], dtype=np.float64).T  # plot fn expects [2,C] or [C,2]
            plot_product_plane(xy, Wrep, p, pfigs / f"{tag}_product.png", f"p={p} {kind} k={k}  h(a,b) colored by ab")
            plot_decision_regions(Wrep, np.array(row["b"]), xy, p, pfigs / f"{tag}_regions.png", f"p={p} {kind} k={k}  readout decision regions")

    for k, group in nonunit_by_k.items():
        xys = [np.array(r["xy"], dtype=np.float64) for r in group]
        plot_aligned_overlay(xys, p, k, pfigs / f"overlay_nonunit_k{k}.png", f"p={p} all non-unit k={k} aligned embeddings")

    return {
        "p": p,
        "n_saved_exact": n_pop,
        "n_population_exact": prev["train"]["n_exact"],
        "population": prev["train"]["population"],
        "kinds": dict(kinds),
        "summary_by_kind": {k: summarize_kind(v) for k, v in by_kind.items()},
        "nonunit_by_k": {str(k): summarize_kind(v) for k, v in nonunit_by_k.items()},
        "descriptor_consistency": desc,
        "polarity_consistency": polarity_consistency,
        "theoretical": theoretical_kernel_card(p),
        "rows": [strip_heavy(r) for r in rows],
        "representatives": [
            {"kind": kind, "k": k, "metrics": strip_heavy(row), "xy": row["xy"], "W_xy": row["W_xy"], "b": row["b"]}
            for kind, k, row, primary in reps
            if primary
        ],
    }


def answers_from(results: dict) -> dict:
    out = {}
    for p, payload in results.items():
        p = int(p)
        nu = payload["summary_by_kind"].get("non_unit", {"n": 0})
        th = payload["theoretical"]
        byk = payload["nonunit_by_k"]
        fibers_txt = {
            k: th["by_k"][k]["fibers"]
            for k in byk
        }
        out[str(p)] = {
            "Q1_gcd_kernel": {k: {"gcd": th["by_k"][k]["gcd"], "kernel_size": th["by_k"][k]["kernel_size"], "n_phases": th["by_k"][k]["n_phases"]} for k in byk},
            "Q2_collapsed_residues": fibers_txt,
            "Q3_radius_distinguishes": {
                "mean_within_fiber_sep": nu.get("mean_within_fiber_sep"),
                "frac_fibers_ratio_gt_1_2": nu.get("frac_fibers_separated"),
            },
            "Q4_consistency": payload["polarity_consistency"],
            "Q5_product_phase_quotient": {
                "mean": nu.get("mean_quotient_acc"),
                "frac_exact": nu.get("frac_quotient_exact"),
            },
            "Q6_product_radius_resolves": {
                "mean_joint_acc": nu.get("mean_joint_acc"),
                "frac_joint_exact": nu.get("frac_joint_exact"),
                "mean_H_given_phase": nu.get("mean_H_target_given_phase"),
                "mean_H_given_phase_radius": nu.get("mean_H_target_given_phase_radius"),
                "frac_bins_disjoint_radii": nu.get("frac_phase_bins_radius_disjoint"),
                "mixed_vs_dlog_parity": nu.get("mean_mixed_parity_acc"),
            },
            "Q7_discrete_decode": {
                "phase_only": nu.get("mean_phase_only_acc"),
                "radius_only": nu.get("mean_radius_only_acc"),
                "joint": nu.get("mean_joint_acc"),
                "expected_phase_only_if_d2": 0.5 if all(th["by_k"][k]["gcd"] == 2 for k in byk) else None,
            },
            "Q8_readout": {
                "unit_normalize_h": nu.get("mean_unit_h_acc"),
                "mean_radius_h": nu.get("mean_meanr_h_acc"),
                "W_angle_spread_within_fiber_deg": nu.get("mean_W_angle_spread_deg"),
                "W_radius_ratio_within_fiber": nu.get("mean_W_radius_ratio"),
            },
            "census": payload["kinds"],
            "revised_metrics": {
                "P_domain_exact": payload["n_population_exact"] / payload["population"],
                "n_domain_exact": payload["n_population_exact"],
                "n_saved": payload["n_saved_exact"],
                "n_faithful_unit_saved": payload["kinds"].get("faithful_unit", 0),
                "n_non_unit_saved": payload["kinds"].get("non_unit", 0),
                "n_ambiguous_saved": payload["kinds"].get("ambiguous", 0),
            },
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--finish-run", type=Path, default=None)
    args = parser.parse_args()
    device = resolve_device(args.device)
    if args.finish_run is not None:
        run_dir = args.finish_run
        print(f"re-analyzing into existing run (hypotheses.json left untouched): {run_dir}")
    else:
        run_dir = new_run_dir(ROOT, "stage_a5_step1")
        locked = locked_payload()
        locked["environment"] = environment_record(str(ROOT))
        write_json(run_dir / "metrics" / "hypotheses.json", locked)
        print(f"run directory: {run_dir}")
        print("Hypotheses locked before analysis:")
        for k, v in HYPOTHESES.items():
            print(f"  {k}: {v}")
        print("Census thresholds:", CENSUS_THRESHOLDS)
        print("Theoretical fibers (group theory, not data):")
        for p, card in locked["theoretical_kernels"].items():
            print(f"  p={p} g={card['g']}")
            for k, info in card["by_k"].items():
                if int(k) % 2 == 0 and int(k) != 0:
                    print(f"    k={k} gcd={info['gcd']} phases={info['n_phases']} fibers={info['fibers']} a,-a={info['fibers_are_a_minus_a']}")

    results = {}
    for p in STAGE_A5_PRIMES:
        results[str(p)] = analyze_prime(p, device, run_dir / "figures")
        write_json(run_dir / "metrics" / f"p{p}.json", {**results[str(p)], "rows": results[str(p)]["rows"]})

    answers = answers_from(results)
    write_json(
        run_dir / "metrics" / "step1.json",
        {
            "classification_note": (
                "An initial gate that required R²>=0.85 for every class put all gcd>1 exact nets "
                "in 'ambiguous'. That gate is methodologically hostile to H1 (variable radius "
                "lowers constant-radius R²). Non-unit membership is now by gcd(k_eff,n)>1 unless "
                "the top two windings straddle unit/non-unit. Hypotheses H1–H5 were not changed."
            ),
            "thresholds": CENSUS_THRESHOLDS,
            "answers": answers,
            "summaries": {p: {k: v for k, v in payload.items() if k != "rows"} for p, payload in results.items()},
        },
    )
    print("\n===== STEP 1 STOP =====")
    print("No ablations, no scramble, no Stage B.")
    print("wrote", run_dir / "metrics" / "step1.json")


if __name__ == "__main__":
    main()
