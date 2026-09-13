#!/usr/bin/env python3
"""Phase 6: when does the F_13* representation appear during training?"""

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

from src.analysis.fourier import discrete_log_table
from src.analysis.group_law import (
    centered_hom_deg_batch,
    fit_dlog_circle_batch,
    fits_from_head_batches,
    restore_params,
    snapshot_params,
    split_accuracies,
    units_mod,
    write_exact_embeddings,
)
from src.config import ExperimentConfig
from src.hardware import resolve_device
from src.repro import environment_record, seed_everything
from src.runner import fit_and_build, prepare_task, train_prepared
from src.storage import new_run_dir, write_json
from src.visualization.group_cert import _save

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

UNITS = set(units_mod(12))


def plot_dynamics(epochs, groups: dict[str, dict[str, np.ndarray]], out: Path, ylabel: str, title: str, ylim=None):
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    colors = {"exact": "#3d5a80", "fail": "#ee6c4d"}
    for name, series in groups.items():
        med = series["median"]
        lo = series["p25"]
        hi = series["p75"]
        ax.plot(epochs, med, color=colors.get(name, "black"), label=name)
        ax.fill_between(epochs, lo, hi, color=colors.get(name, "gray"), alpha=0.2)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend()
    _save(fig, out)


def summarize_group(arr: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
    """arr [T, P]; mask [P]."""
    sub = arr[:, mask]
    if sub.shape[1] == 0:
        z = np.full(arr.shape[0], np.nan)
        return {"median": z, "p25": z, "p75": z}
    return {
        "median": np.median(sub, axis=1),
        "p25": np.percentile(sub, 25, axis=1),
        "p75": np.percentile(sub, 75, axis=1),
    }


def first_epoch(epochs: list[int], flag: np.ndarray) -> np.ndarray:
    """flag [T, P] bool; return epoch of first True, or -1."""
    t, p = flag.shape
    out = np.full(p, -1, dtype=np.int64)
    any_true = flag.any(axis=0)
    first = np.argmax(flag, axis=0)
    out[any_true] = np.array(epochs, dtype=np.int64)[first[any_true]]
    return out


def run_one(cfg: ExperimentConfig, heads: int, device, run_dir: Path) -> dict:
    local = copy.deepcopy(cfg)
    local.model.interaction = "complex"
    local.model.hidden_dims = [heads]
    local.model.activation = "identity"
    local.task.name = "modular_multiplication_star"
    local.task.split = "held_row"
    local.task.held_operands = [12]
    local.run_name = f"dyn_h{heads}"
    seed_everything(cfg.seed)
    prepared = prepare_task(local, device, protocol="split")
    model, fitted = fit_and_build(local, prepared, device, seed=cfg.seed)
    dlog = discrete_log_table(13)
    n = 12
    logs = {
        "epoch": [],
        "train_acc": [],
        "test_acc": [],
        "domain_acc": [],
        "r2": [],
        "k": [],
        "unit_k": [],
        "hom_deg": [],
        "sub_domain_acc": [],
        "sub_test_acc": [],
        "sub_exact": [],
    }

    def on_log(epoch: int, payload: dict) -> None:
        E = payload["model"].E.detach().cpu().numpy()
        xy = E[:, :, 0, :]
        batches = [fit_dlog_circle_batch(E[:, :, h, :], dlog, n) for h in range(heads)]
        batch = batches[0]
        hom = centered_hom_deg_batch(xy, 13)
        snap = snapshot_params(payload["model"])
        from src.analysis.group_law import fits_from_head_batches

        fits = fits_from_head_batches(batches, n, dlog)
        write_exact_embeddings(payload["model"], fits, "aligned")
        acc = split_accuracies(
            payload["model"],
            prepared.domain_x,
            prepared.domain_y,
            prepared.split.train_idx,
            prepared.split.test_idx,
        )
        restore_params(payload["model"], snap)
        logs["epoch"].append(epoch)
        logs["train_acc"].append(payload["train_acc"].numpy())
        logs["test_acc"].append(payload["test_acc"].numpy())
        logs["domain_acc"].append(payload["domain_acc"].numpy())
        logs["r2"].append(batch["r2"])
        logs["k"].append(batch["k"])
        logs["unit_k"].append(np.array([int(k) in UNITS for k in batch["k"]], dtype=np.float64))
        logs["hom_deg"].append(hom)
        logs["sub_domain_acc"].append(acc["domain"].cpu().numpy())
        logs["sub_test_acc"].append(acc["test"].cpu().numpy())
        logs["sub_exact"].append((acc["domain"].cpu().numpy() >= 1.0 - 1e-12).astype(np.float64))

    print(f"H={heads} P={fitted} params={model.n_params_per_network()} train={prepared.train_y.numel()}")
    result = train_prepared(model, prepared, local, show_progress=True, on_log_step=on_log)
    stacked = {k: np.stack(v, axis=0) if k != "epoch" else np.array(v) for k, v in logs.items()}
    exact = result.final_domain_acc.numpy() >= 1.0 - 1e-12
    fail = ~exact
    epochs = stacked["epoch"].tolist()
    order = {
        "r2_ge_0.85": first_epoch(epochs, stacked["r2"] >= 0.85),
        "unit_k": first_epoch(epochs, stacked["unit_k"] >= 0.5),
        "hom_lt_2deg": first_epoch(epochs, stacked["hom_deg"] < 2.0),
        "sub_exact": first_epoch(epochs, stacked["sub_exact"] >= 0.5),
        "domain_exact": first_epoch(epochs, stacked["domain_acc"] >= 1.0 - 1e-12),
        "test_exact": first_epoch(epochs, stacked["test_acc"] >= 1.0 - 1e-12),
    }

    def med_first(mask: np.ndarray, key: str) -> float | None:
        vals = order[key][mask]
        vals = vals[vals >= 0]
        return float(np.median(vals)) if len(vals) else None

    summary = {
        "heads": heads,
        "population": fitted,
        "params": model.n_params_per_network(),
        "n_exact": int(exact.sum()),
        "n_fail": int(fail.sum()),
        "seconds": result.seconds,
        "median_first_epoch_among_eventual_exact": {k: med_first(exact, k) for k in order},
        "frac_eventual_exact_that_hit": {
            k: float((order[k][exact] >= 0).mean()) if exact.any() else None for k in order
        },
    }
    fig_dir = run_dir / "figures"
    groups_exact_fail = lambda arr: {"exact": summarize_group(arr, exact), "fail": summarize_group(arr, fail)}
    plot_dynamics(epochs, groups_exact_fail(stacked["domain_acc"]), fig_dir / f"h{heads}_domain_acc.png", "domain acc", f"H={heads} domain accuracy")
    plot_dynamics(epochs, groups_exact_fail(stacked["test_acc"]), fig_dir / f"h{heads}_test_acc.png", "held-row acc", f"H={heads} held-row accuracy", ylim=(0, 1.05))
    plot_dynamics(epochs, groups_exact_fail(stacked["r2"]), fig_dir / f"h{heads}_r2.png", "dlog-circle R²", f"H={heads} circle R²", ylim=(0, 1.05))
    plot_dynamics(epochs, groups_exact_fail(stacked["hom_deg"]), fig_dir / f"h{heads}_hom.png", "mean |δc| (deg)", f"H={heads} homomorphism residual")
    plot_dynamics(epochs, groups_exact_fail(stacked["unit_k"]), fig_dir / f"h{heads}_unit_k.png", "fraction unit k", f"H={heads} winding is a unit", ylim=(0, 1.05))
    plot_dynamics(epochs, groups_exact_fail(stacked["sub_exact"]), fig_dir / f"h{heads}_sub_exact.png", "fraction sub-exact", f"H={heads} exact-root substitution exact", ylim=(0, 1.05))

    # compact arrays for later (not full T x P of everything in json)
    npz_path = run_dir / "metrics" / f"h{heads}_trajectories.npz"
    np.savez_compressed(
        npz_path,
        epoch=stacked["epoch"],
        domain_acc=stacked["domain_acc"],
        test_acc=stacked["test_acc"],
        train_acc=stacked["train_acc"],
        r2=stacked["r2"],
        k=stacked["k"],
        unit_k=stacked["unit_k"],
        hom_deg=stacked["hom_deg"],
        sub_domain_acc=stacked["sub_domain_acc"],
        sub_exact=stacked["sub_exact"],
        final_exact=exact,
    )
    order_json = {k: v.tolist() for k, v in order.items()}
    write_json(run_dir / "metrics" / f"h{heads}_order.json", {"summary": summary, "first_epoch": order_json})
    print(f"H={heads} exact {int(exact.sum())}/{fitted}")
    print("  median first epoch (eventual exact):", summary["median_first_epoch_among_eventual_exact"])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "held_row.yaml")
    parser.add_argument("--population", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    cfg = ExperimentConfig.from_yaml(args.config)
    if args.population:
        cfg.population.size = args.population
    if args.epochs:
        cfg.train.epochs = args.epochs
    device = resolve_device(cfg.device)
    run_dir = new_run_dir(ROOT, "phase6_dynamics")
    write_json(run_dir / "meta.json", environment_record(str(ROOT)))
    print("run directory:", run_dir)
    summaries = []
    for heads in (1, 3):
        summaries.append(run_one(cfg, heads, device, run_dir))
        write_json(run_dir / "metrics" / "dynamics.json", {"runs": summaries})
    print("wrote", run_dir / "metrics" / "dynamics.json")


if __name__ == "__main__":
    main()
