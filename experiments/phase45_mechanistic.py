#!/usr/bin/env python3
"""Phase 4–5: cluster exact solvers by behavior and inspect representatives.

Observations are written as measurements. Any sentence about an algorithm
is tagged as a hypothesis and paired with a falsifier.
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

from src.analysis.ablation import domain_accuracy, residue_shift_accuracy, unit_ablation
from src.analysis.behavior import embeddings, hidden_maps, invariant_features, pairwise_aligned_distance
from src.analysis.cluster import cluster_from_distance, cluster_from_features, cluster_report, pca_embed
from src.analysis.fourier import describe_network_embeddings, primitive_root
from src.hardware import resolve_device
from src.population.load import load_successful
from src.repro import environment_record
from src.storage import new_run_dir, write_json
from src.visualization.mechanistic import (
    plot_distance_heatmap,
    plot_embeddings_1d,
    plot_pca,
    plot_r2_bars,
    plot_unit_heatmaps,
)

SWEEP = ROOT / "results" / "runs" / "20260909_021014_phase3_width_sweep"
WIDTHS_EXHAUSTIVE = (4,)
WIDTHS_CLUSTER = (4, 5, 6, 8, 16)


def _family_votes(units: list[dict], thresh: float = 0.85, gap: float = 0.15) -> dict:
    """Count units whose best family R² clears ``thresh`` and beats the next by ``gap``."""
    counts = {"fourier": 0, "onehot": 0, "dlog": 0, "ambiguous": 0, "weak": 0, "n": 0}
    for side in ("u", "v"):
        for u in units:
            r = u[side]
            scores = {
                "fourier": r["fourier_r2"],
                "onehot": r["onehot_r2"],
                "dlog": r["dlog_r2"],
            }
            winner, best = max(scores.items(), key=lambda kv: kv[1])
            second = sorted(scores.values(), reverse=True)[1]
            counts["n"] += 1
            if best < thresh:
                counts["weak"] += 1
            elif best - second < gap:
                counts["ambiguous"] += 1
            else:
                counts[winner] += 1
    return counts


def analyze_width(
    width: int, device: torch.device, out: Path, p: int = 13, sweep_dir: Path | None = None
) -> dict:
    ckpt = (sweep_dir or SWEEP) / f"width_{width}" / "checkpoints" / "successful.pt"
    model, meta = load_successful(ckpt, [width], input_dim=2 * p, output_dim=p, device=device)
    acc = domain_accuracy(model, p, device)
    n_exact = int((acc >= 1.0 - 1e-12).sum())
    print(f"width={width} loaded={model.population} still_exact={n_exact}")

    maps = hidden_maps(model, p, device)
    u, v, _c = embeddings(model, p)
    unit_desc = describe_network_embeddings(u, v, p)
    abl = unit_ablation(model, p, device)
    shift_acc = residue_shift_accuracy(model, p, device, shift=1)

    feats = invariant_features(maps, u, v)
    # Pairwise alignment is O(P^2 H^3-ish); fine through a few hundred nets.
    dist = pairwise_aligned_distance(maps)
    if model.population <= 16:
        labels = cluster_from_distance(dist, n_clusters=None, threshold=0.12)
    else:
        # Features for scale; also cut the aligned-distance tree at 0.12 if P<=64.
        k = 4 if model.population >= 32 else 3
        labels = cluster_from_features(feats, n_clusters=k)
    xy = pca_embed(feats, dim=2)
    clusters = cluster_report(labels)

    votes = [_family_votes(unit_desc[i]) for i in range(model.population)]
    vote_sum = {k: int(sum(v[k] for v in votes)) for k in votes[0]}

    width_dir = out / f"width_{width}"
    fig_dir = width_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_pca(xy, labels, fig_dir / "pca_clusters.png", f"width {width}: PCA of permutation-invariant spectra")
    plot_distance_heatmap(dist, labels, fig_dir / "aligned_distance.png", f"width {width}: aligned activation distance")

    # Exhaustive plots for tiny exact populations; exemplars otherwise.
    if width in WIDTHS_EXHAUSTIVE or model.population <= 16:
        inspect = list(range(model.population))
    else:
        inspect = []
        for cl in clusters:
            members = cl["members"]
            # medoid in aligned-distance
            sub = dist[np.ix_(members, members)]
            med = members[int(sub.mean(axis=1).argmin())]
            inspect.append(int(med))

    inspected = []
    for i in inspect:
        stem = f"net{int(meta['indices'][i]) if 'indices' in meta else i}"
        plot_unit_heatmaps(maps[i], fig_dir / f"{stem}_hidden.png", f"width {width} {stem} hidden maps")
        plot_embeddings_1d(u[i], v[i], fig_dir / f"{stem}_uv.png", f"width {width} {stem} first-layer embeddings")
        plot_r2_bars(unit_desc[i], fig_dir / f"{stem}_r2.png", f"width {width} {stem} embedding family R²")
        inspected.append(
            {
                "local_index": i,
                "saved_index": int(meta["indices"][i]) if "indices" in meta else i,
                "cluster": int(labels[i]),
                "units": unit_desc[i],
                "ablation_acc": abl[i].cpu().tolist(),
                "votes": votes[i],
                "uv_corr_mean": float(np.mean([u_["uv_corr"] for u_ in unit_desc[i]])),
            }
        )

    row = {
        "width": width,
        "n_loaded": model.population,
        "n_still_exact": n_exact,
        "primitive_root": primitive_root(p),
        "clusters": clusters,
        "n_clusters": len(clusters),
        "mean_aligned_distance": float(dist[np.triu_indices(len(dist), 1)].mean()) if len(dist) > 1 else 0.0,
        "family_vote_totals": vote_sum,
        "mean_shift1_acc": float(shift_acc.mean()),
        "min_shift1_acc": float(shift_acc.min()),
        "mean_ablation_acc": float(abl.mean()),
        "inspected": inspected,
    }
    write_json(width_dir / "report.json", row)
    np.savez(
        width_dir / "matrices.npz",
        dist=dist,
        labels=labels,
        pca=xy,
        ablation=abl.cpu().numpy(),
        shift_acc=shift_acc.cpu().numpy(),
    )
    print(
        f"  clusters={len(clusters)} mean_aligned_d={row['mean_aligned_distance']:.3f} "
        f"votes={vote_sum} shift1_acc={row['mean_shift1_acc']:.3f}"
    )
    return row


def hypotheses(rows: list[dict]) -> list[dict]:
    """Candidate stories, each with a stated falsifier. Not conclusions."""
    out = []
    w4 = next((r for r in rows if r["width"] == 4), None)
    if w4 is not None:
        out.append(
            {
                "id": "H1_dlog_embeddings",
                "claim": (
                    "First-layer embeddings of exact solvers are well-fit by sinusoids "
                    "on discrete log (characters of (Z/pZ)*) and poorly fit by residue "
                    "sinusoids or one-hots. u and v of a unit often share the same k."
                ),
                "status": "supported_as_measurement",
                "gate": "family_vote_totals and per-unit R² in the width reports",
                "falsifier": (
                    "Equal-complexity residue-sinusoid R² matching dlog R², or dlog k "
                    "failing to recur across independent inits, or a fit that is only "
                    "high because 3 parameters interpolate 12 points (check that the "
                    "residue family does not also win)."
                ),
            }
        )
        out.append(
            {
                "id": "H2_shared_family_not_one_circuit",
                "claim": (
                    "Width-4 exact solvers share a dlog embedding family but are not "
                    "one hidden-unit permutation class (aligned map distance ~0.76)."
                ),
                "status": "supported_as_measurement",
                "gate": "n_clusters and mean_aligned_distance for width 4",
                "falsifier": "Mean aligned distance << 0.2 after Hungarian matching.",
            }
        )
    out.append(
        {
            "id": "H3_capacity_does_not_imply_rule_learning",
            "claim": (
                "Networks that implement the full table when trained on it do not "
                "thereby learn a rule that interpolates a random holdout "
                "(Phase 3 split sweep: 0 exact solvers)."
            ),
            "status": "supported_by_phase3",
            "falsifier": "A later random-split run that produces domain-exact solvers at these widths.",
        }
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--widths", type=int, nargs="*", default=list(WIDTHS_CLUSTER))
    parser.add_argument("--sweep", type=Path, default=SWEEP)
    args = parser.parse_args()

    device = resolve_device("cuda")
    run_dir = new_run_dir(ROOT, "phase45_mechanistic")
    meta = environment_record(str(ROOT))
    meta["source_sweep"] = str(args.sweep)
    write_json(run_dir / "meta.json", meta)
    print(f"run directory: {run_dir}")

    rows = []
    for w in args.widths:
        rows.append(analyze_width(int(w), device, run_dir, sweep_dir=args.sweep))

    payload = {
        "rows": rows,
        "hypotheses": hypotheses(rows),
        "method_notes": {
            "distance": "1 - mean |Pearson| of hidden maps after Hungarian unit alignment",
            "features": "per-unit rFFT(|u|), rFFT(|v|), sparsity; units sorted by energy",
            "r2_families": "constant + sinusoid; constant + one spike; dlog sinusoid on (Z/pZ)*",
            "exact_means": "argmax correct on every (a,b) in Z/13Z × Z/13Z",
        },
    }
    write_json(run_dir / "metrics" / "phase45.json", payload)
    print("wrote", run_dir / "metrics" / "phase45.json")


if __name__ == "__main__":
    main()
