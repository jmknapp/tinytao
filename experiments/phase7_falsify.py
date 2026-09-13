#!/usr/bin/env python3
"""Phase 7: falsifiers for the discrete-log embedding story.

Stated falsifier
----------------
If first-layer embeddings that win the dlog family (R² ≥ 0.85) keep a
high dlog R² after *additive* or *random* relabelings of the residue
axis, the 3-parameter sinusoid is too flexible and the multiplicative-
group measurement is not about group structure.

The story survives this particular test only if:
  - multiplicative relabeling a ↦ c·a (0 fixed) keeps dlog R² high, and
  - additive a ↦ a+k and random permutations of F_p* drop it.

A second measurement, not a proof: hidden maps factored through
(a·b), (a+b), dlog(a)+dlog(b), and the zero pattern.
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

from src.analysis.ablation import domain_accuracy
from src.analysis.behavior import embeddings, hidden_maps
from src.analysis.falsify import collect_embeddings, hidden_factor_r2, orbit_r2
from src.analysis.fourier import describe_embedding
from src.hardware import resolve_device
from src.population.load import load_successful
from src.repro import environment_record
from src.storage import new_run_dir, write_json
from src.visualization.phase7 import plot_factor_bars, plot_orbit_bars

SOURCES = [
    {
        "name": "width3",
        "width": 3,
        "path": ROOT
        / "results"
        / "runs"
        / "20260909_022909_phase6_minimal"
        / "width3_hunt"
        / "checkpoints"
        / "successful.pt",
    },
    {
        "name": "width4",
        "width": 4,
        "path": ROOT
        / "results"
        / "runs"
        / "20260909_021014_phase3_width_sweep"
        / "width_4"
        / "checkpoints"
        / "successful.pt",
    },
    {
        "name": "width8",
        "width": 8,
        "path": ROOT
        / "results"
        / "runs"
        / "20260909_021014_phase3_width_sweep"
        / "width_8"
        / "checkpoints"
        / "successful.pt",
    },
]


def _mean_pack(orbits: list[dict], which: str, high_only: bool, thresh: float = 0.85) -> dict:
    chosen = []
    for o in orbits:
        if high_only and o["base"]["dlog_r2"] < thresh:
            continue
        chosen.append(o[which])
    if not chosen:
        return {
            "n": 0,
            "mean_dlog": None,
            "min_dlog": None,
            "max_dlog": None,
            "mean_fourier": None,
            "frac_dlog_ge_085": None,
            "frac_fourier_ge_085": None,
        }
    return {
        "n": len(chosen),
        "mean_dlog": float(np.mean([c["mean_dlog"] for c in chosen])),
        "min_dlog": float(np.min([c["min_dlog"] for c in chosen])),
        "max_dlog": float(np.max([c["max_dlog"] for c in chosen])),
        "mean_fourier": float(np.mean([c["mean_fourier"] for c in chosen])),
        "frac_dlog_ge_085": float(np.mean([c["frac_dlog_ge_085"] for c in chosen])),
        "frac_fourier_ge_085": float(np.mean([c["frac_fourier_ge_085"] for c in chosen])),
    }


def analyze_source(spec: dict, device: torch.device, p: int, rng: np.random.Generator) -> dict:
    model, meta = load_successful(spec["path"], [spec["width"]], 2 * p, p, device=device)
    acc = domain_accuracy(model, p, device)
    n_exact = int((acc >= 1.0 - 1e-12).sum())
    print(f"{spec['name']}: loaded={model.population} still_exact={n_exact}")

    u, v, _c = embeddings(model, p)
    maps = hidden_maps(model, p, device).cpu().numpy()
    vecs = collect_embeddings(u, v)
    orbits = [orbit_r2(y, p, rng) for y in vecs]
    bases = [o["base"]["dlog_r2"] for o in orbits]
    n_high = int(sum(r >= 0.85 for r in bases))

    factors = []
    for i in range(maps.shape[0]):
        for h in range(maps.shape[1]):
            factors.append(hidden_factor_r2(maps[i, h], p))
    factor_mean = {k: float(np.mean([f[k] for f in factors])) for k in factors[0]}

    unit_rows = []
    idx = 0
    for i in range(model.population):
        for h in range(spec["width"]):
            unit_rows.append(
                {
                    "net": i,
                    "unit": h,
                    "u": describe_embedding(u[i, h], p),
                    "v": describe_embedding(v[i, h], p),
                    "orbit_u": orbits[idx],
                    "orbit_v": orbits[idx + 1],
                    "hidden_factor": factors[i * spec["width"] + h],
                }
            )
            idx += 2
            # describe_embedding includes spectrum; drop it in the saved unit table
            unit_rows[-1]["u"].pop("spectrum", None)
            unit_rows[-1]["v"].pop("spectrum", None)

    high = {
        "n_sides": n_high,
        "n_sides_total": len(orbits),
        "base_dlog": float(np.mean([b for b in bases if b >= 0.85])) if n_high else None,
        "multiplicative": _mean_pack(orbits, "multiplicative", high_only=True),
        "additive": _mean_pack(orbits, "additive", high_only=True),
        "random": _mean_pack(orbits, "random", high_only=True),
    }
    all_sides = {
        "n_sides": len(orbits),
        "base_dlog": float(np.mean(bases)),
        "base_fourier": float(np.mean([o["base"]["fourier_r2"] for o in orbits])),
        "multiplicative": _mean_pack(orbits, "multiplicative", high_only=False),
        "additive": _mean_pack(orbits, "additive", high_only=False),
        "random": _mean_pack(orbits, "random", high_only=False),
    }

    # Width-3 unit 1 is the unclassified unit; keep its factors visible.
    special = None
    if spec["width"] == 3 and model.population == 1:
        special = {
            "unit_1_factor": factors[1],
            "unit_1_u_dlog": float(unit_rows[1]["u"]["dlog_r2"]),
            "unit_1_u_onehot": float(unit_rows[1]["u"]["onehot_r2"]),
            "unit_1_v_dlog": float(unit_rows[1]["v"]["dlog_r2"]),
            "unit_1_v_onehot": float(unit_rows[1]["v"]["onehot_r2"]),
        }

    print(
        f"  high-dlog sides {n_high}/{len(orbits)}  "
        f"orbit dlog mean  ×c={high['multiplicative']['mean_dlog']}  "
        f"+k={high['additive']['mean_dlog']}  rand={high['random']['mean_dlog']}"
    )
    print(
        f"  hidden factor   product={factor_mean['product']:.3f}  "
        f"sum={factor_mean['sum']:.3f}  dlog_sum={factor_mean['dlog_sum']:.3f}  "
        f"zero={factor_mean['zero_pattern']:.3f}"
    )

    return {
        "name": spec["name"],
        "width": spec["width"],
        "n": model.population,
        "n_exact": n_exact,
        "indices": meta["indices"].tolist() if torch.is_tensor(meta.get("indices")) else None,
        "high_dlog": high,
        "all_sides": all_sides,
        "hidden_factor": factor_mean,
        "special": special,
        "units": unit_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    device = resolve_device("cuda")
    p = 13
    rng = np.random.default_rng(args.seed)
    run_dir = new_run_dir(ROOT, "phase7_falsify")
    write_json(run_dir / "meta.json", environment_record(str(ROOT)))
    print(f"run directory: {run_dir}")

    rows = []
    for spec in SOURCES:
        row = analyze_source(spec, device, p, rng)
        write_json(run_dir / "metrics" / f"{spec['name']}.json", row)
        rows.append(row)

    summary = [{k: r[k] for k in ("name", "width", "n", "n_exact", "high_dlog", "all_sides", "hidden_factor", "special")} for r in rows]
    write_json(run_dir / "metrics" / "falsify.json", {"rows": summary})
    plot_orbit_bars(rows, run_dir / "figures" / "orbit_dlog.png")
    plot_factor_bars(rows, run_dir / "figures" / "hidden_factor.png")
    print("wrote", run_dir / "metrics" / "falsify.json")


if __name__ == "__main__":
    main()
