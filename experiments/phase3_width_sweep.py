#!/usr/bin/env python3
"""Phase 3: Pr(exact solver) versus hidden width / parameter count."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.analysis.stats import wilson_interval
from src.config import ExperimentConfig
from src.hardware import resolve_device
from src.repro import environment_record, seed_everything
from src.runner import finalize_population, fit_and_build, prepare_task, train_prepared
from src.storage import new_run_dir, save_run_skeleton, write_json
from src.visualization.sweep import plot_width_sweep


def _median_first_exact(domain_acc: torch.Tensor, log_epochs: list[int]) -> float | None:
    reached = domain_acc >= (1.0 - 1e-12)
    first: list[int] = []
    t = log_epochs
    for i in range(reached.shape[0]):
        hits = torch.where(reached[i])[0]
        if hits.numel():
            first.append(int(t[int(hits[0])]))
    if not first:
        return None
    return float(torch.tensor(first, dtype=torch.float32).median())


def run_width(cfg: ExperimentConfig, hidden: int, device: torch.device, parent: Path) -> dict:
    local = copy.deepcopy(cfg)
    depth = int(cfg.sweep.depths[0]) if cfg.sweep.depths else 1
    local.model.hidden_dims = [hidden] * max(1, depth)
    local.run_name = f"h{hidden}"

    seed_everything(cfg.seed)
    prepared = prepare_task(local, device, protocol=cfg.sweep.protocol)
    model, fitted = fit_and_build(local, prepared, device, seed=cfg.seed)
    width_dir = parent / f"width_{hidden}"
    for sub in ("metrics", "checkpoints", "figures", "logs"):
        (width_dir / sub).mkdir(parents=True, exist_ok=True)
    save_run_skeleton(width_dir, local, {"fitted_population": fitted, "hidden": hidden})

    print(
        f"width={hidden} params={model.n_params_per_network()} P={fitted} "
        f"protocol={prepared.protocol} epochs={local.train.epochs}"
    )
    result = train_prepared(model, prepared, local, show_progress=True)
    packed = finalize_population(model, result, prepared, width_dir, write_figures=True)
    counts = packed["counts"]
    n = counts.total
    p_hat, lo, hi = wilson_interval(counts.exact, n)
    row = {
        "hidden": hidden,
        "hidden_dims": list(local.model.hidden_dims),
        "params_per_network": model.n_params_per_network(),
        "population": fitted,
        "protocol": prepared.protocol,
        "counts": counts.as_dict(),
        "fractions": counts.fractions(),
        "n_exact": counts.exact,
        "frac_exact": p_hat,
        "frac_exact_lo": lo,
        "frac_exact_hi": hi,
        "mean_train_acc": float(result.final_train_acc.mean()),
        "mean_test_acc": float(result.final_test_acc.mean()),
        "mean_domain_acc": float(result.final_domain_acc.mean()),
        "median_domain_acc": float(result.final_domain_acc.median()),
        "max_domain_acc": float(result.final_domain_acc.max()),
        "n_sudden_transition": int(packed["sudden"].sum()),
        "median_first_exact_epoch": _median_first_exact(result.domain_acc, result.log_epochs),
        "seconds": result.seconds,
        "peak_vram_gib": result.peak_vram_bytes / (1024**3),
        "subdir": str(width_dir.relative_to(parent)),
    }
    write_json(width_dir / "metrics" / "sweep_row.json", row)
    print(
        f"  exact {counts.exact}/{n} ({p_hat:.3f} [{lo:.3f},{hi:.3f}])  "
        f"mean_domain={row['mean_domain_acc']:.3f} max={row['max_domain_acc']:.3f}"
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 hidden-width sweep")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "phase3_width_sweep.yaml")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--widths", type=int, nargs="*", default=None)
    parser.add_argument("--protocol", type=str, default=None, help="full_domain or split")
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    if args.protocol:
        cfg.sweep.protocol = args.protocol
    if args.widths:
        cfg.sweep.hidden_widths = list(args.widths)
    if args.smoke:
        cfg.population.size = min(cfg.population.size, 64)
        cfg.train.epochs = min(cfg.train.epochs, 80)
        cfg.train.log_every = 10
        cfg.train.eval_every = 10
        cfg.sweep.hidden_widths = cfg.sweep.hidden_widths[:3]
        cfg.run_name = f"{cfg.run_name}_smoke"

    device = resolve_device(cfg.device)
    run_dir = new_run_dir(ROOT, cfg.run_name)
    meta = environment_record(str(ROOT))
    meta["sweep"] = {
        "hidden_widths": cfg.sweep.hidden_widths,
        "protocol": cfg.sweep.protocol,
        "depths": cfg.sweep.depths,
        "population": cfg.population.size,
        "epochs": cfg.train.epochs,
    }
    save_run_skeleton(run_dir, cfg, meta)
    print(f"run directory: {run_dir}")
    print(f"device: {device}  protocol: {cfg.sweep.protocol}")

    rows = []
    for hidden in cfg.sweep.hidden_widths:
        rows.append(run_width(cfg, int(hidden), device, run_dir))
        write_json(run_dir / "metrics" / "sweep.json", {"rows": rows})

    figures = plot_width_sweep(rows, run_dir / "figures")
    n_exact_total = sum(r["n_exact"] for r in rows)
    smallest = None
    for r in rows:
        if r["n_exact"] > 0:
            smallest = {"hidden": r["hidden"], "params": r["params_per_network"], "n_exact": r["n_exact"]}
            break
    write_json(
        run_dir / "metrics" / "sweep.json",
        {
            "rows": rows,
            "n_exact_total": n_exact_total,
            "smallest_exact_architecture": smallest,
            "figures": [str(p) for p in figures],
        },
    )
    print("smallest exact architecture:", smallest)
    print("figures:")
    for p in figures:
        print(" ", p)


if __name__ == "__main__":
    main()
