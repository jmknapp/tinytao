#!/usr/bin/env python3
"""Phase 6b: shrink known exact solvers and hunt for anything smaller than width 4."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.analysis.ablation import domain_accuracy, unit_ablation
from src.analysis.shrink import copy_kept_units, least_critical_units, quantize_inplace
from src.analysis.stats import wilson_interval
from src.config import ExperimentConfig, ModelConfig, PopulationConfig, TaskConfig, TrainConfig
from src.hardware import resolve_device
from src.population.load import load_successful
from src.repro import environment_record, seed_everything
from src.runner import finalize_population, fit_and_build, prepare_task, train_prepared
from src.storage import new_run_dir, write_json
from src.visualization.phase6 import plot_quantize

W4_CKPT = (
    ROOT
    / "results"
    / "runs"
    / "20260909_021014_phase3_width_sweep"
    / "width_4"
    / "checkpoints"
    / "successful.pt"
)


def quantize_curve(model, p: int, device, bits_list: list[int]) -> list[dict]:
    rows = []
    w1 = [t.detach().clone() for t in model.weights]
    b1 = [t.detach().clone() for t in model.biases] if model.biases is not None else None
    for bits in bits_list:
        with torch.no_grad():
            for t, src in zip(model.weights, w1):
                t.copy_(src)
            if b1 is not None:
                for t, src in zip(model.biases, b1):
                    t.copy_(src)
        quantize_inplace(model, bits)
        acc = domain_accuracy(model, p, device)
        n_exact = int((acc >= 1.0 - 1e-12).sum())
        rows.append(
            {
                "bits": bits,
                "n_exact": n_exact,
                "n": model.population,
                "mean_domain": float(acc.mean()),
                "min_domain": float(acc.min()),
            }
        )
        print(f"  quantize {bits}-bit: exact {n_exact}/{model.population} mean_acc={acc.mean():.3f}")
    with torch.no_grad():
        for t, src in zip(model.weights, w1):
            t.copy_(src)
        if b1 is not None:
            for t, src in zip(model.biases, b1):
                t.copy_(src)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width3-population", type=int, default=4096)
    parser.add_argument("--width3-epochs", type=int, default=12000)
    parser.add_argument("--skip-width3-hunt", action="store_true")
    args = parser.parse_args()

    device = resolve_device("cuda")
    p = 13
    run_dir = new_run_dir(ROOT, "phase6_minimal")
    write_json(run_dir / "meta.json", environment_record(str(ROOT)))
    print(f"run directory: {run_dir}")

    model4, meta4 = load_successful(W4_CKPT, [4], 2 * p, p, device=device)
    acc0 = domain_accuracy(model4, p, device)
    print(f"loaded width-4 exact solvers: {int((acc0 >= 1).sum())}/{model4.population}")

    print("quantization...")
    qrows = quantize_curve(model4, p, device, bits_list=[16, 8, 6, 5, 4, 3, 2])
    plot_quantize(
        [r["bits"] for r in qrows],
        [r["n_exact"] for r in qrows],
        model4.population,
        run_dir / "figures" / "quantize_width4.png",
    )

    print("prune least-critical unit -> width 3...")
    abl = unit_ablation(model4, p, device)
    keep = least_critical_units(abl, n_drop=1)
    pruned = copy_kept_units(model4, keep, device)
    acc_pruned = domain_accuracy(pruned, p, device)
    n_exact_pruned = int((acc_pruned >= 1).sum())
    print(f"  pruned (no retrain): exact {n_exact_pruned}/{pruned.population} max={float(acc_pruned.max()):.3f}")

    print("retrain pruned width-3 inits on the full table...")
    seed_everything(0)
    cfg_re = ExperimentConfig(
        task=TaskConfig(name="modular_multiplication", p=p),
        model=ModelConfig(hidden_dims=[3]),
        population=PopulationConfig(size=pruned.population, min_size=1),
        train=TrainConfig(epochs=8000, lr=3e-3, log_every=100, eval_every=100),
        seed=0,
    )
    prepared = prepare_task(cfg_re, device, protocol="full_domain")
    result_re = train_prepared(pruned, prepared, cfg_re, show_progress=True)
    packed_re = finalize_population(pruned, result_re, prepared, run_dir / "pruned_retrain", write_figures=True)
    print(f"  pruned+retrain exact {packed_re['counts'].exact}/{packed_re['counts'].total}")

    hunt = None
    if not args.skip_width3_hunt:
        print(f"width-3 from-scratch hunt P={args.width3_population} epochs={args.width3_epochs}...")
        seed_everything(1)
        cfg_h = ExperimentConfig(
            task=TaskConfig(name="modular_multiplication", p=p),
            model=ModelConfig(hidden_dims=[3]),
            population=PopulationConfig(size=args.width3_population, min_size=64),
            train=TrainConfig(epochs=args.width3_epochs, lr=5e-3, log_every=100, eval_every=100),
            seed=1,
        )
        prepared_h = prepare_task(cfg_h, device, protocol="full_domain")
        model3, fitted = fit_and_build(cfg_h, prepared_h, device, seed=1)
        result_h = train_prepared(model3, prepared_h, cfg_h, show_progress=True)
        packed_h = finalize_population(model3, result_h, prepared_h, run_dir / "width3_hunt", write_figures=True)
        ph, lo, hi = wilson_interval(packed_h["counts"].exact, packed_h["counts"].total)
        hunt = {
            "population": fitted,
            "epochs": args.width3_epochs,
            "lr": 5e-3,
            "counts": packed_h["counts"].as_dict(),
            "frac_exact": ph,
            "frac_exact_lo": lo,
            "frac_exact_hi": hi,
            "max_domain_acc": float(result_h.final_domain_acc.max()),
            "mean_domain_acc": float(result_h.final_domain_acc.mean()),
            "params": model3.n_params_per_network(),
        }
        print(
            f"  hunt exact {packed_h['counts'].exact}/{fitted} ({ph:.4f}) "
            f"max_domain={hunt['max_domain_acc']:.3f}"
        )

    smallest = {
        "architecture": "width 4, float32",
        "params": model4.n_params_per_network(),
        "n_found": model4.population,
        "source": "phase3 full-domain sweep",
    }
    still_exact_bits = [r["bits"] for r in qrows if r["n_exact"] > 0]
    if still_exact_bits:
        qmin = min(still_exact_bits)
        qn = next(r["n_exact"] for r in qrows if r["bits"] == qmin)
        if qn:
            smallest_q = {
                "architecture": f"width 4, {qmin}-bit quantized",
                "params": model4.n_params_per_network(),
                "bits": qmin,
                "n_found": qn,
            }
        else:
            smallest_q = None
    else:
        smallest_q = None
    if packed_re["counts"].exact > 0:
        smallest = {
            "architecture": "width 3 (pruned from width 4 + retrain)",
            "params": pruned.n_params_per_network(),
            "n_found": packed_re["counts"].exact,
        }
    if hunt and hunt["counts"]["exact"] > 0:
        smallest = {
            "architecture": "width 3 from scratch",
            "params": hunt["params"],
            "n_found": hunt["counts"]["exact"],
        }

    report = {
        "width4_loaded": model4.population,
        "quantize": qrows,
        "prune_no_retrain": {
            "n_exact": n_exact_pruned,
            "max_domain": float(acc_pruned.max()),
            "mean_domain": float(acc_pruned.mean()),
            "params": pruned.n_params_per_network(),
        },
        "prune_retrain": packed_re["counts"].as_dict(),
        "width3_hunt": hunt,
        "smallest_exact": smallest,
        "smallest_quantized_exact": smallest_q,
        "saved_indices": meta4["indices"].tolist() if "indices" in meta4 else None,
    }
    write_json(run_dir / "metrics" / "minimal.json", report)
    print("smallest exact:", smallest)
    print("smallest quantized exact:", smallest_q)


if __name__ == "__main__":
    main()
