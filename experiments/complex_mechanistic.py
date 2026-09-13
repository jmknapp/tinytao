#!/usr/bin/env python3
"""Mechanistic probes on the smallest Model C generalizers (H=1, random 70/30)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.analysis.complex_mech import analyze_complex_checkpoint
from src.hardware import resolve_device
from src.storage import write_json

DEFAULT_CKPT = (
    ROOT
    / "results"
    / "runs"
    / "20260909_104121_complex_composition"
    / "complex_h1_random_70"
    / "checkpoints"
    / "successful.pt"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    device = resolve_device("cuda")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    report = analyze_complex_checkpoint(ckpt, device)
    report["checkpoint"] = str(args.checkpoint)
    out = args.out or args.checkpoint.parents[1] / "metrics" / "mechanistic.json"
    write_json(out, report)
    print(f"loaded {report['n_loaded']} exact H=1 nets, identity exact {report['identity_n_exact']}")
    print(
        f"circle R^2 mean={report['mean_circle_r2']:.3f} median={report['median_circle_r2']:.3f} "
        f"frac>=0.85={report['frac_r2_ge_0.85']:.3f} frac>=0.95={report['frac_r2_ge_0.95']:.3f}"
    )
    print(f"k counts {report['k_counts']}")
    print(
        f"homomorphism mean |e(a)e(b)-e(ab)|={report['mean_hom_product_abs_err']:.4f} "
        f"mean angle err={report['mean_hom_angle_err_rad']:.4f} rad"
    )
    sub = report["substitute_fitted_circle"]
    print(
        f"substitute fitted circle, freeze readout: exact {sub['n_exact']}/{sub['n']} "
        f"mean domain={sub['mean_domain_acc']:.3f} max={sub['max_domain_acc']:.3f}"
    )
    print("wrote", out)


if __name__ == "__main__":
    main()
