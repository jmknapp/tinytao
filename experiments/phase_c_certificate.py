#!/usr/bin/env python3
"""Phases 1–4 on saved Model C held-row generalizers (p=13).

No retraining. Causal tests of the F_13* representation hypothesis.
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.analysis.fourier import discrete_log_table, primitive_root
from src.analysis.group_law import (
    apply_scale_rot,
    collect_head0_xy,
    exact_counts,
    exact_unit_circle,
    fit_all_heads,
    homomorphism_delta,
    load_population,
    lstsq_readout,
    pair_masks,
    permute_embedding_by_power,
    perturb_angle,
    radius_stats,
    random_permute_embeddings,
    restore_params,
    rotate_embeddings,
    rotate_readout_for_embedding_rotation,
    snapshot_params,
    split_accuracies,
    summarize_delta,
    unit_normalize_learned,
    units_mod,
    winding_readout,
    wrap,
    write_exact_embeddings,
)
from src.hardware import resolve_device
from src.repro import environment_record
from src.storage import new_run_dir, write_json
from src.tasks.base import SplitKind
from src.tasks.modular import ModularMultiplicationStar
from src.visualization.group_cert import heatmap, hist_deg

HELD_ROW_RUN = ROOT / "results" / "runs" / "20260909_104844_held_row"
P = 13
HELD = 12


def _task_tensors(device: torch.device):
    task = ModularMultiplicationStar(p=P)
    a, b, x, y = task.full_domain(device)
    split = task.split(SplitKind.HELD_ROW, seed=0, held_operands=[HELD])
    return task, a, b, x, y, split


def _eval(model, x, y, split) -> dict:
    acc = split_accuracies(model, x, y, split.train_idx, split.test_idx)
    return {k: exact_counts(v) for k, v in acc.items()}


def _agg(values: list[dict], key: str) -> dict[str, float]:
    arr = np.array([v[key] for v in values], dtype=np.float64)
    return {"mean": float(arr.mean()), "median": float(np.median(arr)), "max": float(arr.max()), "min": float(arr.min())}


def phase12(E_cpu: torch.Tensor, dlog: np.ndarray, train_mask: np.ndarray) -> dict:
    n = E_cpu.shape[1]
    xy = collect_head0_xy(E_cpu)
    pop = xy.shape[0]
    deltas = np.stack([homomorphism_delta(xy[i], P) for i in range(pop)])
    summaries = [summarize_delta(deltas[i], train_mask) for i in range(pop)]
    fits = fit_all_heads(E_cpu, dlog)
    k0 = np.array([f[0]["k"] for f in fits])
    r2 = np.array([f[0]["r2"] for f in fits])
    conj = np.array([bool(f[0]["conjugate"]) for f in fits])
    units = set(units_mod(n))
    r2_all = np.array([[fh["r2"] for fh in f] for f in fits])
    k_all = np.array([[fh["k"] for fh in f] for f in fits])
    rad = [radius_stats(xy[i], dlog, n) for i in range(pop)]
    centered = np.stack([wrap(deltas[i] - summaries[i]["circ_mean_rad"]) for i in range(pop)])
    mean_abs_c = np.abs(centered).mean(axis=0)
    order = np.argsort([int(dlog[i + 1]) for i in range(n)])
    k_counts = {int(k): int((k0 == k).sum()) for k in sorted(set(k0.tolist()))}
    return {
        "n_networks": pop,
        "heads": int(E_cpu.shape[2]),
        "primitive_root": int(primitive_root(P)),
        "delta_head0": {
            "mean_abs_raw_rad": _agg(summaries, "mean_abs_raw_rad"),
            "mean_abs_centered_rad": _agg(summaries, "mean_abs_centered_rad"),
            "mean_abs_centered_deg": _agg(summaries, "mean_abs_centered_deg"),
            "max_abs_centered_deg": _agg(summaries, "max_abs_centered_deg"),
            "rms_centered_rad": _agg(summaries, "rms_centered_rad"),
            "mean_abs_centered_train_rad": _agg(summaries, "mean_abs_centered_train_rad"),
            "mean_abs_centered_held_rad": _agg(summaries, "mean_abs_centered_held_rad"),
            "frac_max_centered_lt_5deg": float((np.array([s["max_abs_centered_deg"] for s in summaries]) < 5.0).mean()),
            "frac_max_centered_lt_2deg": float((np.array([s["max_abs_centered_deg"] for s in summaries]) < 2.0).mean()),
        },
        "radius_head0": {
            "mean": _agg(rad, "mean"),
            "cv": _agg(rad, "cv"),
            "min": _agg(rad, "min"),
            "max": _agg(rad, "max"),
            "corr_residue": _agg(rad, "corr_residue"),
            "corr_dlog": _agg(rad, "corr_dlog"),
        },
        "fit_head0": {
            "mean_r2": float(r2.mean()),
            "min_r2": float(r2.min()),
            "median_r2": float(np.median(r2)),
            "k_counts": k_counts,
            "frac_unit_k": float(np.mean([int(k) in units for k in k0])),
            "n_non_unit": int(sum(int(k) not in units for k in k0)),
            "frac_conjugate": float(conj.mean()),
            "units_mod_n": sorted(units),
        },
        "fit_all_heads": {
            "mean_r2": float(r2_all.mean()),
            "min_r2": float(r2_all.min()),
            "k_counts": {int(k): int((k_all == k).sum()) for k in sorted(set(k_all.reshape(-1).tolist()))},
            "frac_unit_k": float(np.mean([int(k) in units for k in k_all.reshape(-1)])),
        },
        "heatmaps": {
            "mean_abs_centered_residue": mean_abs_c,
            "mean_abs_centered_dlog": mean_abs_c[np.ix_(order, order)],
            "dlog_order_residues": [int(i + 1) for i in order],
        },
        "per_net_max_centered_deg": np.array([s["max_abs_centered_deg"] for s in summaries]),
        "per_net_mean_centered_deg": np.array([s["mean_abs_centered_deg"] for s in summaries]),
        "fits": fits,
        "k0": k0,
        "r2": r2,
    }


def run_substitutions(model, snap, fits, x, y, split) -> dict:
    variants = {}
    restore_params(model, snap)
    variants["E_learned_frozen_W"] = _eval(model, x, y, split)

    restore_params(model, snap)
    write_exact_embeddings(model, fits, "fitted")
    variants["D_fitted_sinusoid_frozen_W"] = _eval(model, x, y, split)

    restore_params(model, snap)
    write_exact_embeddings(model, fits, "aligned")
    variants["A_aligned_exact_roots_frozen_W"] = _eval(model, x, y, split)

    restore_params(model, snap)
    write_exact_embeddings(model, fits, "strict_unit")
    variants["A_strict_unit_roots_frozen_W"] = _eval(model, x, y, split)

    restore_params(model, snap)
    write_exact_embeddings(model, fits, "strict_unit")
    lstsq_readout(model, x, y)
    variants["B_strict_unit_roots_LS_W"] = _eval(model, x, y, split)

    restore_params(model, snap)
    unit_normalize_learned(model)
    variants["C_unit_normalized_learned_frozen_W"] = _eval(model, x, y, split)

    restore_params(model, snap)
    with torch.no_grad():
        rmean = model.E.norm(dim=-1).mean(dim=(1, 2))
        unit_normalize_learned(model)
        model.W_out.mul_(rmean.reshape(model.population, 1, 1) ** 2)
    variants["C_unit_norm_E_scale_matched_W"] = _eval(model, x, y, split)

    restore_params(model, snap)
    write_exact_embeddings(model, fits, "aligned")
    dlog = discrete_log_table(P)
    n = model.n_group
    with torch.no_grad():
        for i, net_fits in enumerate(fits):
            # symbolic: aligned exact embeddings + geometric nearest-root readout
            if model.heads == 1:
                W, b = winding_readout(net_fits[0], dlog, n)
                model.W_out[i] = torch.as_tensor(W, device=model.W_out.device, dtype=model.W_out.dtype)
                model.b_out[i] = torch.as_tensor(b, device=model.b_out.device, dtype=model.b_out.dtype)
    variants["symbolic_aligned_roots_geometric_W"] = _eval(model, x, y, split)
    restore_params(model, snap)
    return variants


def run_interventions(model, snap, fits, x, y, split, k0: np.ndarray) -> dict:
    out: dict[str, Any] = {}
    dlog = discrete_log_table(P)
    n = model.n_group

    # Global rotations
    rot_rows = []
    for deg in (0, 15, 30, 45, 90, 180):
        phi = math.radians(deg)
        restore_params(model, snap)
        rotate_embeddings(model, phi)
        frozen = _eval(model, x, y, split)
        restore_params(model, snap)
        rotate_embeddings(model, phi)
        rotate_readout_for_embedding_rotation(model, phi)
        adjusted = _eval(model, x, y, split)
        rot_rows.append({"deg": deg, "phi": phi, "frozen_W": frozen, "predicted_W_adjust": adjusted})
    out["global_rotation"] = rot_rows

    # Radius already in substitutions; repeat named
    restore_params(model, snap)
    unit_normalize_learned(model)
    out["radius_normalization"] = _eval(model, x, y, split)

    # Winding swap k=1 <-> k=5
    def _swap(src_k: int, dst_k: int) -> dict:
        restore_params(model, snap)
        swapped = copy.deepcopy(fits)
        n_src = 0
        for i, net_fits in enumerate(swapped):
            if int(net_fits[0]["k"]) != src_k or model.heads != 1:
                continue
            n_src += 1
            f = dict(net_fits[0])
            f["k"] = dst_k
            unit = exact_unit_circle(dst_k, dlog, n, bool(f["conjugate"]))
            f["fitted"] = apply_scale_rot(unit, float(f["scale"]), float(f["phi"]))
            swapped[i][0] = f
        restore_params(model, snap)
        write_exact_embeddings(model, swapped, "aligned")
        no_w = _eval(model, x, y, split)
        # predicted readout for those nets only; others stay identity (already restored then overwritten E)
        restore_params(model, snap)
        write_exact_embeddings(model, swapped, "aligned")
        with torch.no_grad():
            for i, net_fits in enumerate(swapped):
                if int(fits[i][0]["k"]) != src_k:
                    continue
                W, b = winding_readout(net_fits[0], dlog, n)
                model.W_out[i] = torch.as_tensor(W, device=model.W_out.device, dtype=model.W_out.dtype)
                model.b_out[i] = torch.as_tensor(b, device=model.b_out.device, dtype=model.b_out.dtype)
        yes_w = _eval(model, x, y, split)
        return {"n_source": n_src, "without_W_adjust": no_w, "with_predicted_W": yes_w}

    if model.heads == 1:
        out["winding_1_to_5"] = _swap(1, 5)
        out["winding_5_to_1"] = _swap(5, 1)
        autos = []
        for q in units_mod(P - 1):
            restore_params(model, snap)
            permute_embedding_by_power(model, q, P)
            frozen = _eval(model, x, y, split)
            restore_params(model, snap)
            with torch.no_grad():
                for i, net_fits in enumerate(fits):
                    f = dict(net_fits[0])
                    f["k"] = (int(f["k"]) * q) % (P - 1)
                    if f["k"] == 0:
                        f["k"] = P - 1
                    W, b = winding_readout(f, dlog, n)
                    unit = exact_unit_circle(int(f["k"]), dlog, n, bool(f["conjugate"]))
                    xy = apply_scale_rot(unit, float(f["scale"]), float(f["phi"]))
                    model.E[i, :, 0, 0] = torch.as_tensor(xy[:, 0], device=model.E.device, dtype=model.E.dtype)
                    model.E[i, :, 0, 1] = torch.as_tensor(xy[:, 1], device=model.E.device, dtype=model.E.dtype)
                    model.W_out[i] = torch.as_tensor(W, device=model.W_out.device, dtype=model.W_out.dtype)
                    model.b_out[i] = torch.as_tensor(b, device=model.b_out.device, dtype=model.b_out.dtype)
            adjusted = _eval(model, x, y, split)
            autos.append(
                {
                    "q": q,
                    "gcd_ok": math.gcd(q, P - 1) == 1,
                    "permute_E_frozen_W": frozen,
                    "predicted_new_winding_E_and_W": adjusted,
                }
            )
        out["automorphisms"] = autos
    else:
        out["winding_1_to_5"] = {"skipped": "heads>1"}
        out["winding_5_to_1"] = {"skipped": "heads>1"}
        out["automorphisms"] = {"skipped": "heads>1"}
    pert = []
    a_cpu = (torch.arange(1, P).repeat_interleave(P - 1)).cpu().numpy()
    b_cpu = (torch.arange(1, P).repeat(P - 1)).cpu().numpy()
    for residue, deg in ((12, 15), (12, 45), (1, 15), (3, 15)):
        restore_params(model, snap)
        perturb_angle(model, residue, math.radians(deg))
        logits = model(x)
        pred = logits.argmax(dim=-1).cpu()
        y_cpu = y.cpu()
        wrong = pred != y_cpu.unsqueeze(1)
        # fraction of nets wrong on each pair
        pair_frac = wrong.float().mean(dim=1).numpy()
        affect = (a_cpu == residue) | (b_cpu == residue)
        pert.append(
            {
                "residue": residue,
                "deg": deg,
                "eval": _eval(model, x, y, split),
                "mean_error_rate_affected_pairs": float(pair_frac[affect].mean()),
                "mean_error_rate_unaffected_pairs": float(pair_frac[~affect].mean()),
                "max_error_rate_unaffected": float(pair_frac[~affect].max()),
                "pair_error_rate": pair_frac,
            }
        )
    out["angle_perturbation"] = pert

    restore_params(model, snap)
    random_permute_embeddings(model, seed=0)
    out["random_permutation_frozen_W"] = _eval(model, x, y, split)
    restore_params(model, snap)
    return out


def _jsonify(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items() if k not in {"fits", "heatmaps", "pair_error_rate", "per_net_max_centered_deg", "per_net_mean_centered_deg", "k0", "r2"}}
    if isinstance(obj, list):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def analyze_checkpoint(path: Path, device: torch.device, x, y, split, dlog, train_mask, figures: Path, tag: str) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = load_population(ckpt, device)
    snap = snapshot_params(model)
    print(f"{tag}: loaded {model.population} nets, heads={model.heads}, params={model.n_params_per_network()}")
    identity = _eval(model, x, y, split)
    print(f"  identity domain exact {identity['domain']['n_exact']}/{identity['domain']['n']}")

    p12 = phase12(ckpt["E"], dlog, train_mask)
    print(
        f"  circle R^2 mean={p12['fit_head0']['mean_r2']:.4f} min={p12['fit_head0']['min_r2']:.4f} "
        f"unit k frac={p12['fit_head0']['frac_unit_k']:.3f} k={p12['fit_head0']['k_counts']}"
    )
    print(
        f"  centered |delta| mean={p12['delta_head0']['mean_abs_centered_deg']['mean']:.3f} deg "
        f"held={np.degrees(p12['delta_head0']['mean_abs_centered_held_rad']['mean']):.3f} deg "
        f"max mean={p12['delta_head0']['max_abs_centered_deg']['mean']:.3f} deg"
    )

    n = P - 1
    residues = [str(i) for i in range(1, P)]
    heatmap(
        p12["heatmaps"]["mean_abs_centered_residue"],
        figures / f"{tag}_delta_residue.png",
        fr"{tag}: mean |centered $\delta$| (rad), residue axes",
        "b",
        "a",
        residues,
        residues,
    )
    dlab = [str(r) for r in p12["heatmaps"]["dlog_order_residues"]]
    heatmap(
        p12["heatmaps"]["mean_abs_centered_dlog"],
        figures / f"{tag}_delta_dlog.png",
        fr"{tag}: mean |centered $\delta$| (rad), dlog order",
        "b (by dlog)",
        "a (by dlog)",
        dlab,
        dlab,
    )
    hist_deg(p12["per_net_max_centered_deg"], figures / f"{tag}_delta_max_hist.png", f"{tag}: per-net max |centered delta|")

    restore_params(model, snap)
    subs = run_substitutions(model, snap, p12["fits"], x, y, split)
    for name, ev in subs.items():
        print(f"  sub {name}: domain exact {ev['domain']['n_exact']}/{ev['domain']['n']} mean={ev['domain']['mean_acc']:.3f} test exact {ev['test']['n_exact']}")

    restore_params(model, snap)
    inter = run_interventions(model, snap, p12["fits"], x, y, split, p12["k0"])
    for row in inter["global_rotation"]:
        print(
            f"  rot {row['deg']:>3} deg  frozen exact {row['frozen_W']['domain']['n_exact']}  "
            f"W-adjust exact {row['predicted_W_adjust']['domain']['n_exact']}"
        )
    w15 = inter["winding_1_to_5"]
    w51 = inter["winding_5_to_1"]
    if "skipped" not in w15:
        print(
            f"  winding 1->5 without W {w15['without_W_adjust']['domain']['n_exact']} "
            f"with predicted W {w15['with_predicted_W']['domain']['n_exact']}"
        )
        print(
            f"  winding 5->1 without W {w51['without_W_adjust']['domain']['n_exact']} "
            f"with predicted W {w51['with_predicted_W']['domain']['n_exact']}"
        )
    for row in inter["angle_perturbation"]:
        print(
            f"  perturb a={row['residue']} +{row['deg']}deg  affected {row['mean_error_rate_affected_pairs']:.3f} "
            f"unaffected {row['mean_error_rate_unaffected_pairs']:.4f} domain exact {row['eval']['domain']['n_exact']}"
        )
        err = row["pair_error_rate"].reshape(n, n)
        heatmap(
            err,
            figures / f"{tag}_perturb_a{row['residue']}_d{row['deg']}.png",
            f"{tag}: pair error rate after +{row['deg']} deg on {row['residue']}",
            "b",
            "a",
            residues,
            residues,
        )
    if "skipped" not in inter["automorphisms"]:
        for row in inter["automorphisms"]:
            print(
                f"  auto q={row['q']} permute-E frozen-W exact {row['permute_E_frozen_W']['domain']['n_exact']} "
                f"predicted E+W exact {row['predicted_new_winding_E_and_W']['domain']['n_exact']}"
            )
    print(f"  random perm frozen-W exact {inter['random_permutation_frozen_W']['domain']['n_exact']}")

    report = {
        "checkpoint": str(path),
        "tag": tag,
        "identity": identity,
        "phase12": _jsonify(p12),
        "substitutions": _jsonify(subs),
        "interventions": _jsonify(inter),
        "params_per_network": model.n_params_per_network(),
        "population_saved": model.population,
        "heads": model.heads,
    }
    restore_params(model, snap)
    return report


def analyze_failures(path: Path, device, x, y, split, dlog, train_mask) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    labels = ckpt["labels"].cpu().numpy()
    keep = labels != 4
    if keep.sum() == 0:
        return {"n_failed": 0}
    sub = {"E": ckpt["E"][keep], "W_out": ckpt["W_out"][keep], "b_out": ckpt["b_out"][keep]}
    model = load_population(sub, device)
    p12 = phase12(sub["E"], dlog, train_mask)
    acc = _eval(model, x, y, split)
    return {
        "n_failed": int(keep.sum()),
        "label_counts": {int(l): int((labels[keep] == l).sum()) for l in sorted(set(labels[keep].tolist()))},
        "identity": acc,
        "delta_head0": p12["delta_head0"],
        "fit_head0": p12["fit_head0"],
        "radius_head0": p12["radius_head0"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--held-row-run", type=Path, default=HELD_ROW_RUN)
    args = parser.parse_args()
    device = resolve_device("cuda")
    run_dir = new_run_dir(ROOT, "phase_c_certificate")
    write_json(run_dir / "meta.json", environment_record(str(ROOT)))
    task, a, b, x, y, split = _task_tensors(device)
    dlog = discrete_log_table(P)
    masks = pair_masks(P, HELD, "a")
    print(f"run directory: {run_dir}")
    print(f"p={P} g={primitive_root(P)} train={int(split.train_idx.numel())} test={int(split.test_idx.numel())}")

    h1 = args.held_row_run / "complex_h1_held_row" / "checkpoints" / "successful.pt"
    h3 = args.held_row_run / "complex_h3_held_row" / "checkpoints" / "successful.pt"
    rep = args.held_row_run / "complex_h1_held_row" / "checkpoints" / "representatives.pt"

    reports = {
        "protocol": {
            "p": P,
            "primitive_root": int(primitive_root(P)),
            "task": "modular_multiplication_star",
            "split": "held_row",
            "held_residue": HELD,
            "n_train": int(split.train_idx.numel()),
            "n_test": int(split.test_idx.numel()),
            "n_domain": 144,
            "n_train_classes": 12,
            "n_test_classes": 12,
            "optimizer": "adam",
            "lr": 0.003,
            "epochs": 8000,
            "seed": 0,
            "source_run": str(args.held_row_run),
            "git_commit": None,
            "note": "workspace is not a git repository; commit unavailable",
        }
    }
    reports["h1_held_row"] = analyze_checkpoint(h1, device, x, y, split, dlog, masks["train"], run_dir / "figures", "h1_held_row")
    reports["h3_held_row"] = analyze_checkpoint(h3, device, x, y, split, dlog, masks["train"], run_dir / "figures", "h3_held_row")
    reports["h1_held_row_failures"] = analyze_failures(rep, device, x, y, split, dlog, masks["train"])
    print("failures", reports["h1_held_row_failures"].get("n_failed"), reports["h1_held_row_failures"].get("fit_head0"))
    write_json(run_dir / "metrics" / "certificate.json", reports)
    print("wrote", run_dir / "metrics" / "certificate.json")


if __name__ == "__main__":
    main()
