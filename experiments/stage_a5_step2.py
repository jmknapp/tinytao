#!/usr/bin/env python3
"""Stage A.5 Step 2: causal tests of quotient + radial binary code on p=11.

Writes preregistration BEFORE evaluating any modified network.
Uses saved exact H=1 checkpoints only. No retraining, scramble, Stage B, or p=7.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.analysis.fourier import discrete_log_table
from src.analysis.group_law import fit_dlog_circle_batch, load_population, lstsq_readout, pair_masks, units_mod
from src.analysis.level4_h1 import effective_winding
from src.analysis.radial_causal import (
    apply_exact_phases,
    apply_global_level_swap,
    apply_two_level_radii,
    apply_unit_norm,
    change_metrics,
    classify_table,
    dlog_vals,
    is_exact,
    mask_metrics,
    n_in_fiber,
    pair_accuracy,
    permute_radii,
    polar_parts,
    predict_single_fiber_labels,
    predict_subset_labels,
    residue_grid,
    rho_parity,
    swap_fiber_radii,
    symbolic_decode,
)
from src.analysis.radial_quotient import classify_winding, fiber_id_from_k, fiber_members, winding_scores
from src.analysis.stage_a5_hypotheses import (
    CENSUS_THRESHOLDS,
    H_RADIAL_QUOTIENT_V2,
    HYPOTHESES,
    PREREGISTRATION,
    STAGE_A_RUN,
)
from src.population.complex import PopulationComplexMLP
from src.repro import environment_record
from src.storage import new_run_dir, write_json
from src.tasks.modular import ModularMultiplicationStar
from src.visualization.stage_a5_step2 import grouped_exact_bar, heatmap_bool, hist_values

P = 11
N = 10
STAGE_A_DIR = ROOT / STAGE_A_RUN
CKPT = STAGE_A_DIR / "complex_h1_p11_held_row" / "checkpoints" / "successful.pt"


def load_nonunit_cpu() -> dict:
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = load_population(ckpt, torch.device("cpu"))
    dlog = discrete_log_table(P)
    E = model.E.detach().cpu()[:, :, 0, :].numpy().astype(np.float64)
    W = model.W_out.detach().cpu().numpy().astype(np.float64)
    B = model.b_out.detach().cpu().numpy().astype(np.float64)
    batch = fit_dlog_circle_batch(E, dlog, N)
    k_eff = effective_winding(batch["k"], batch["conjugate"], N)
    scores = winding_scores(E, dlog, N)
    units = set(units_mod(N))
    keep = []
    for i in range(E.shape[0]):
        order = np.argsort(scores["r2_k"][i])[::-1]
        k2 = int(order[1])
        kind = classify_winding(
            int(k_eff[i]),
            N,
            float(batch["r2"][i]),
            float(scores["r2_k"][i, k2]),
            k2 in units,
            CENSUS_THRESHOLDS["r2_clear"],
            CENSUS_THRESHOLDS["r2_gap"],
        )
        if kind == "non_unit":
            keep.append(i)
    idx = np.array(keep, dtype=np.int64)
    return {
        "xy": E[idx],
        "W": W[idx],
        "b": B[idx],
        "k": k_eff[idx].astype(np.int64),
        "phi": batch["phi"][idx],
        "r2": batch["r2"][idx],
        "n": int(len(idx)),
        "saved_indices": idx.tolist(),
    }


def ls_readout_acc(xy: np.ndarray, domain_x: torch.Tensor, domain_y: torch.Tensor, target: np.ndarray) -> dict:
    pop = xy.shape[0]
    n = xy.shape[1]
    model = PopulationComplexMLP(pop, 2 * n, [1], n, device="cpu")
    with torch.no_grad():
        model.E[:, :, 0, :] = torch.as_tensor(xy, dtype=torch.float32)
        model.W_out.zero_()
        model.b_out.zero_()
    lstsq_readout(model, domain_x.float(), domain_y)
    W = model.W_out.detach().cpu().numpy()
    B = model.b_out.detach().cpu().numpy()
    exact = 0
    accs = []
    for i in range(pop):
        pred = classify_table(xy[i], W[i], B[i])
        accs.append(pair_accuracy(pred, target))
        exact += int(is_exact(pred, target))
    return {"n_exact": exact, "mean_acc": float(np.mean(accs)), "frac_exact": exact / pop}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finish-run", type=Path, default=None)
    args = parser.parse_args()
    if args.finish_run is None:
        run_dir = new_run_dir(ROOT, "stage_a5_step2")
        locked = {
            "step": 2,
            "p": P,
            "H_RADIAL_QUOTIENT_V2": H_RADIAL_QUOTIENT_V2,
            "preregistration": PREREGISTRATION,
            "prior_hypotheses": HYPOTHESES,
            "class2_criterion": PREREGISTRATION["8_class2_criterion"],
            "not_in_this_step": ["p=7 interventions", "scrambled controls", "Stage B", "retraining"],
            "environment": environment_record(str(ROOT)),
            "source_checkpoint": str(CKPT),
        }
        write_json(run_dir / "metrics" / "preregistration.json", locked)
        print(f"run directory: {run_dir}")
        print("H_RADIAL_QUOTIENT_V2 locked.")
        print(H_RADIAL_QUOTIENT_V2)
        print("\nPreregistered predictions:")
        for k, v in PREREGISTRATION.items():
            print(f"  {k}: {v}")
        print("\nLoading embeddings after preregistration write.")
    else:
        run_dir = args.finish_run
        print(f"re-using run dir {run_dir} (preregistration must already exist)")

    pop = load_nonunit_cpu()
    n_net = pop["n"]
    print(f"non-unit exact nets: {n_net}")
    dv = dlog_vals(P)
    s = dv % 2
    aa, bb, target = residue_grid(P)
    train_mask = pair_masks(P, P - 1, "a")["train"]
    fid_by_k = {}
    for k in (2, 4, 6, 8):
        fid_by_k[k] = fiber_id_from_k(k, dv, N)

    orig_pred = np.stack([classify_table(pop["xy"][i], pop["W"][i], pop["b"][i]) for i in range(n_net)])
    assert all(is_exact(orig_pred[i], target) for i in range(n_net))

    figs = run_dir / "figures"

    # ---- Experiment 1: global level swap ----
    e1 = []
    for i in range(n_net):
        xy2 = apply_global_level_swap(pop["xy"][i], s)
        pred = classify_table(xy2, pop["W"][i], pop["b"][i])
        row = mask_metrics(pred, orig_pred[i], train_mask, dv, target)
        row["label_agree_with_original"] = pair_accuracy(pred, orig_pred[i])
        row["n_labels_changed"] = int((pred != orig_pred[i]).sum())
        even = (dv[target - 1] % 2) == 0
        row["n_changed_even_cells"] = int(((pred != orig_pred[i]) & even).sum())
        row["n_changed_odd_cells"] = int(((pred != orig_pred[i]) & ~even).sum())
        e1.append(row)
    e1_exact = int(sum(r["exact"] for r in e1))
    print(f"E1 global level swap: exact {e1_exact}/{n_net}  mean acc {np.mean([r['domain'] for r in e1]):.3f}  "
          f"odd-parity acc {np.mean([r['odd_product_parity'] for r in e1]):.3f}  "
          f"even-parity acc {np.mean([r['even_product_parity'] for r in e1]):.3f}")

    # ---- Experiment 2+3: every fiber, every net ----
    e23_rows = []
    f1s = []
    label_agree = []
    for i in range(n_net):
        k = int(pop["k"][i])
        fibers = fiber_members(k, P)
        for f, members in fibers.items():
            u = members[0]
            n_in = n_in_fiber(aa, bb, u, P)
            pred_changed = n_in == 1
            pred_labels = predict_single_fiber_labels(orig_pred[i], n_in, P)
            xy2 = swap_fiber_radii(pop["xy"][i], u, P)
            actual = classify_table(xy2, pop["W"][i], pop["b"][i])
            actual_changed = actual != orig_pred[i]
            cm = change_metrics(pred_changed, actual_changed)
            agr = pair_accuracy(actual, pred_labels)
            e23_rows.append(
                {
                    "net": i,
                    "k": k,
                    "fiber": int(f),
                    "u": int(u),
                    **cm,
                    "predicted_label_agreement": agr,
                    "exact_after": is_exact(actual, target),
                    "double_cells_changed": int(((n_in == 2) & actual_changed).sum()),
                    "double_cells_label_agree": float((actual[n_in == 2] == pred_labels[n_in == 2]).mean()),
                    "one_cells_label_agree": float((actual[n_in == 1] == pred_labels[n_in == 1]).mean()),
                    "zero_cells_label_agree": float((actual[n_in == 0] == pred_labels[n_in == 0]).mean()),
                }
            )
            f1s.append(cm["f1"])
            label_agree.append(agr)
            if i == 0 and f == 0:
                flab = "{" + ",".join(str(m) for m in members) + "}"
                heatmap_bool(pred_changed, figs / "ex_pred_changed_fiber0.png", f"Predicted changed cells, net0 fiber {flab}", "b", "a")
                heatmap_bool(actual_changed, figs / "ex_actual_changed_fiber0.png", f"Actual changed cells, net0 fiber {flab}", "b", "a")
                heatmap_bool(actual != pred_labels, figs / "ex_label_mismatch_fiber0.png", "Predicted vs actual label mismatch", "b", "a")

    print(
        f"E2/3 single-fiber: mean F1 {np.mean(f1s):.3f}  mean predicted-label agreement {np.mean(label_agree):.3f}  "
        f"frac F1==1 {np.mean([x >= 1 - 1e-12 for x in f1s]):.3f}  "
        f"frac label-exact {np.mean([x >= 1 - 1e-12 for x in label_agree]):.3f}"
    )
    hist_values(np.array(f1s), figs / "hist_single_fiber_f1.png", "F1 (predicted vs actual changed cells)", "p=11 single-fiber radius swap")
    hist_values(np.array(label_agree), figs / "hist_single_fiber_label_agree.png", "predicted-label agreement", "p=11 single-fiber predicted permutation")

    # ---- Experiment 4: subset swaps ----
    e4 = {"one": e23_rows, "two": [], "half3": [], "all": []}
    fiber_ids = list(range(5))
    pairs = [(0, 1), (0, 2), (1, 3), (2, 4)]
    for i in range(n_net):
        k = int(pop["k"][i])
        fid = fid_by_k[k]
        orig = orig_pred[i]
        # two fibers
        for s0, s1 in pairs:
            swapped = {s0, s1}
            pred_lab, flip = predict_subset_labels(orig, aa, bb, swapped, fid, P)
            xy2 = pop["xy"][i].copy()
            r, th = polar_parts(xy2)
            for f in swapped:
                u = fiber_members(k, P)[f][0]
                iu, iv = u - 1, (P - u) - 1
                r[iu], r[iv] = r[iv], r[iu]
            xy2 = np.stack([r * np.cos(th), r * np.sin(th)], axis=-1)
            actual = classify_table(xy2, pop["W"][i], pop["b"][i])
            e4["two"].append(
                {
                    "agreement": pair_accuracy(actual, pred_lab),
                    "change_f1": change_metrics(flip, actual != orig)["f1"],
                    "exact": is_exact(actual, target),
                }
            )
        # three fibers (random half rounded up)
        swapped = {0, 2, 4}
        pred_lab, flip = predict_subset_labels(orig, aa, bb, swapped, fid, P)
        r, th = polar_parts(pop["xy"][i])
        r = r.copy()
        for f in swapped:
            u = fiber_members(k, P)[f][0]
            iu, iv = u - 1, (P - u) - 1
            r[iu], r[iv] = r[iv], r[iu]
        xy2 = np.stack([r * np.cos(th), r * np.sin(th)], axis=-1)
        actual = classify_table(xy2, pop["W"][i], pop["b"][i])
        e4["half3"].append(
            {
                "agreement": pair_accuracy(actual, pred_lab),
                "change_f1": change_metrics(flip, actual != orig)["f1"],
                "exact": is_exact(actual, target),
            }
        )
        # all fibers = pairwise even/odd exchange
        swapped = set(fiber_ids)
        pred_lab, flip = predict_subset_labels(orig, aa, bb, swapped, fid, P)
        r, th = polar_parts(pop["xy"][i])
        r = r.copy()
        for f in swapped:
            u = fiber_members(k, P)[f][0]
            iu, iv = u - 1, (P - u) - 1
            r[iu], r[iv] = r[iv], r[iu]
        xy2 = np.stack([r * np.cos(th), r * np.sin(th)], axis=-1)
        actual = classify_table(xy2, pop["W"][i], pop["b"][i])
        e4["all"].append(
            {
                "agreement": pair_accuracy(actual, pred_lab),
                "change_f1": change_metrics(flip, actual != orig)["f1"],
                "exact": is_exact(actual, target),
                "n_changed": int((actual != orig).sum()),
                "mean_acc": pair_accuracy(actual, target),
            }
        )
    for name in ("two", "half3", "all"):
        agr = np.mean([r["agreement"] for r in e4[name]])
        f1 = np.mean([r["change_f1"] for r in e4[name]])
        print(f"E4 subset {name}: mean label agreement {agr:.3f}  mean change F1 {f1:.3f}")

    # ---- Experiments 5–8 substitutions ----
    task = ModularMultiplicationStar(p=P)
    _, _, domain_x, domain_y = task.full_domain("cpu")

    def eval_xy(make_xy, tag_decoder=False):
        exact_w = 0
        accs = []
        exact_sym = 0
        sym_accs = []
        for i in range(n_net):
            xy2 = make_xy(i)
            pred = classify_table(xy2, pop["W"][i], pop["b"][i])
            accs.append(pair_accuracy(pred, target))
            exact_w += int(is_exact(pred, target))
            r, _ = polar_parts(xy2)
            rho_e, rho_o = rho_parity(r, s)
            # if two-level already, rho from the embedding
            dec = symbolic_decode(xy2, int(pop["k"][i]), float(pop["phi"][i]), dv, P, rho_e, rho_o)
            sym_accs.append(pair_accuracy(dec, target))
            exact_sym += int(is_exact(dec, target))
        return {
            "frozen_W_n_exact": exact_w,
            "frozen_W_frac": exact_w / n_net,
            "frozen_W_mean_acc": float(np.mean(accs)),
            "symbolic_decoder_n_exact": exact_sym,
            "symbolic_decoder_frac": exact_sym / n_net,
            "symbolic_decoder_mean_acc": float(np.mean(sym_accs)),
        }

    e5 = eval_xy(lambda i: apply_exact_phases(pop["xy"][i], int(pop["k"][i]), float(pop["phi"][i]), dv, N))
    print(f"E5 exact phases + learned r, frozen W: exact {e5['frozen_W_n_exact']}/{n_net} acc {e5['frozen_W_mean_acc']:.3f}  "
          f"decoder {e5['symbolic_decoder_n_exact']}/{n_net}")

    def two_level_learned_phase(i):
        r, _ = polar_parts(pop["xy"][i])
        rho_e, rho_o = rho_parity(r, s)
        return apply_two_level_radii(pop["xy"][i], s, rho_e, rho_o)

    e6a = eval_xy(two_level_learned_phase)
    print(f"E6a two radii + learned phases, frozen W: exact {e6a['frozen_W_n_exact']}/{n_net}  "
          f"decoder {e6a['symbolic_decoder_n_exact']}/{n_net}")

    def two_level_exact_phase(i):
        r, _ = polar_parts(pop["xy"][i])
        rho_e, rho_o = rho_parity(r, s)
        xy2 = apply_two_level_radii(pop["xy"][i], s, rho_e, rho_o)
        return apply_exact_phases(xy2, int(pop["k"][i]), float(pop["phi"][i]), dv, N)

    e6b = eval_xy(two_level_exact_phase)
    print(f"E6b two radii + exact phases, frozen W: exact {e6b['frozen_W_n_exact']}/{n_net}  "
          f"decoder {e6b['symbolic_decoder_n_exact']}/{n_net}")

    # Experiment 7: rho_even=1, only ratio, scale W by rho_even^2
    e7_exact = 0
    e7_acc = []
    ratios = []
    for i in range(n_net):
        r, _ = polar_parts(pop["xy"][i])
        rho_e, rho_o = rho_parity(r, s)
        q = rho_o / (rho_e + 1e-12)
        ratios.append(q)
        xy2 = apply_two_level_radii(pop["xy"][i], s, 1.0, q)
        xy2 = apply_exact_phases(xy2, int(pop["k"][i]), float(pop["phi"][i]), dv, N)
        Wsc = pop["W"][i] * (rho_e**2)
        pred = classify_table(xy2, Wsc, pop["b"][i] * 0.0)  # bias not scaled with h; try scaled W only keep b
        # h scales as rho^2 relative to unit-even; original h ~ rho_e^2 * unit_even product
        # z_new = z_old / rho_e, h_new = h_old / rho_e^2, so W_new = W_old * rho_e^2 restores logits if b=0
        pred_b = classify_table(xy2, Wsc, pop["b"][i])
        e7_acc.append(pair_accuracy(pred_b, target))
        e7_exact += int(is_exact(pred_b, target))
    e7 = {
        "n_exact_Wscale_keep_b": e7_exact,
        "mean_acc": float(np.mean(e7_acc)),
        "mean_rho_ratio_odd_over_even": float(np.mean(ratios)),
        "std_rho_ratio": float(np.std(ratios)),
    }
    print(f"E7 ratio-only + W*rho_even^2: exact {e7_exact}/{n_net} acc {e7['mean_acc']:.3f}  mean q={e7['mean_rho_ratio_odd_over_even']:.3f}")

    # Experiment 8 already in eval_xy symbolic_decoder on e6b embedding
    e8 = e6b

    # Experiment 10 unit-norm
    e10_frozen = eval_xy(lambda i: apply_unit_norm(pop["xy"][i]))
    xy_unit = np.stack([apply_unit_norm(pop["xy"][i]) for i in range(n_net)])
    e10_ls = ls_readout_acc(xy_unit, domain_x, domain_y, target)
    print(f"E10 unit-norm frozen W: exact {e10_frozen['frozen_W_n_exact']}/{n_net} acc {e10_frozen['frozen_W_mean_acc']:.3f}")
    print(f"E10 unit-norm LS readout: exact {e10_ls['n_exact']}/{n_net} acc {e10_ls['mean_acc']:.3f}  (bound 0.5 pair / 0 exact)")

    # Experiment 11 radius scramble among residues (not Stage A table-scramble).
    rng = np.random.default_rng(1)
    even_idx = np.where(s == 0)[0]
    odd_idx = np.where(s == 1)[0]
    n_perm = 32
    dest_exact = []
    dest_acc = []
    pres_exact = []
    pres_acc = []
    for _t in range(n_perm):
        de = 0
        da = []
        pe = 0
        pa = []
        perm_d = rng.permutation(N)
        perm_p = np.arange(N)
        perm_p[even_idx] = rng.permutation(even_idx)
        perm_p[odd_idx] = rng.permutation(odd_idx)
        for i in range(n_net):
            xy_d = permute_radii(pop["xy"][i], perm_d)
            xy_p = permute_radii(pop["xy"][i], perm_p)
            pd_ = classify_table(xy_d, pop["W"][i], pop["b"][i])
            pp_ = classify_table(xy_p, pop["W"][i], pop["b"][i])
            da.append(pair_accuracy(pd_, target))
            pa.append(pair_accuracy(pp_, target))
            de += int(is_exact(pd_, target))
            pe += int(is_exact(pp_, target))
        dest_exact.append(de)
        dest_acc.append(float(np.mean(da)))
        pres_exact.append(pe)
        pres_acc.append(float(np.mean(pa)))
    e11 = {
        "n_perm": n_perm,
        "destroying_mean_n_exact": float(np.mean(dest_exact)),
        "destroying_mean_acc": float(np.mean(dest_acc)),
        "preserving_mean_n_exact": float(np.mean(pres_exact)),
        "preserving_mean_acc": float(np.mean(pres_acc)),
        "note": "Radius permutations among embeddings. Stage A multiplication-table scramble is not run.",
    }
    print(
        f"E11 radius permute ({n_perm} perms): destroying exact {e11['destroying_mean_n_exact']:.1f}/{n_net} acc {e11['destroying_mean_acc']:.3f}  "
        f"parity-preserving exact {e11['preserving_mean_n_exact']:.1f}/{n_net} acc {e11['preserving_mean_acc']:.3f}"
    )

    grouped_exact_bar(
        ["identity", "global swap", "exact phase", "2-radius", "2-radius+phase", "unit-norm frozen", "unit-norm LS"],
        [
            n_net,
            e1_exact,
            e5["frozen_W_n_exact"],
            e6a["frozen_W_n_exact"],
            e6b["frozen_W_n_exact"],
            e10_frozen["frozen_W_n_exact"],
            e10_ls["n_exact"],
        ],
        n_net,
        figs / "substitution_exact.png",
        "p=11 non-unit: domain-exact count after interventions (frozen W unless noted)",
    )
    grouped_exact_bar(
        ["frozen W + 2-r + phase", "symbolic decoder on that embedding"],
        [e6b["frozen_W_n_exact"], e6b["symbolic_decoder_n_exact"]],
        n_net,
        figs / "class2_symbolic.png",
        "Class 2 tests: compact embedding vs embedding+decoder",
    )

    n_label_exact = int(sum(1 for r in e23_rows if r["predicted_label_agreement"] >= 1 - 1e-12))
    n_f1_exact = int(sum(1 for r in e23_rows if r["f1"] >= 1 - 1e-12))
    class2_A = e6b["frozen_W_n_exact"] == n_net or e5["frozen_W_n_exact"] == n_net or e6a["frozen_W_n_exact"] == n_net
    class2_B = e6b["symbolic_decoder_n_exact"] == n_net
    # also decoder on learned embedding
    e8_learned = eval_xy(lambda i: pop["xy"][i])
    if e8_learned["symbolic_decoder_n_exact"] == n_net:
        class2_B = True
    population_class = "CLASS 2" if class2_B else ("CLASS 2-A only" if class2_A else "CLASS 1")
    if class2_B:
        population_class = "CLASS 2 — ALTERNATIVE SYMBOLIC SOLVER"
    elif class2_A:
        population_class = "CLASS 2-A (symbolic embedding + frozen W; decoder incomplete)"
    else:
        population_class = "CLASS 1 — DOMAIN-EXACT NON-SYMBOLIC"

    payload = {
        "n_non_unit": n_net,
        "class_population": population_class,
        "class2_A": bool(class2_A),
        "class2_B": bool(class2_B),
        "E1_global_level_swap": {
            "n_exact": e1_exact,
            "mean": {k: float(np.mean([r[k] for r in e1 if not isinstance(r[k], bool)])) for k in e1[0] if k != "exact"},
            "frac_exact": e1_exact / n_net,
        },
        "E2_E3_single_fiber": {
            "n_tests": len(e23_rows),
            "mean_f1": float(np.mean(f1s)),
            "mean_predicted_label_agreement": float(np.mean(label_agree)),
            "frac_f1_exact": n_f1_exact / len(e23_rows),
            "frac_label_agreement_exact": n_label_exact / len(e23_rows),
            "mean_double_cells_changed": float(np.mean([r["double_cells_changed"] for r in e23_rows])),
            "mean_one_cells_label_agree": float(np.mean([r["one_cells_label_agree"] for r in e23_rows])),
            "mean_zero_cells_label_agree": float(np.mean([r["zero_cells_label_agree"] for r in e23_rows])),
            "mean_double_cells_label_agree": float(np.mean([r["double_cells_label_agree"] for r in e23_rows])),
        },
        "E4_subsets": {
            name: {
                "mean_agreement": float(np.mean([r["agreement"] for r in rows])),
                "mean_f1": float(np.mean([r["change_f1"] for r in rows])),
                "n": len(rows),
            }
            for name, rows in e4.items()
            if name != "one"
        },
        "E5_exact_phase": e5,
        "E6a_two_radius_learned_phase": e6a,
        "E6b_two_radius_exact_phase": e6b,
        "E7_ratio_gauge": e7,
        "E8_symbolic_decoder_on_learned": e8_learned,
        "E10_unit_norm": {"frozen": e10_frozen, "lstsq": e10_ls, "theoretical_max_pair_acc": 0.5, "theoretical_max_exact": 0},
        "E11_radius_permute": e11,
    }
    write_json(run_dir / "metrics" / "step2.json", payload)
    write_json(run_dir / "metrics" / "e23_rows.json", {"rows": e23_rows})
    print("\n===== STEP 2 STOP =====")
    print("population class:", population_class)
    print("No scramble, no p=7, no Stage B.")
    print("wrote", run_dir / "metrics" / "step2.json")


if __name__ == "__main__":
    main()
