#!/usr/bin/env python3
"""Phase 5: extract a torch-free symbolic multiplier from saved H=1 nets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.analysis.complex_mech import fit_dlog_circle
from src.analysis.fourier import discrete_log_table, primitive_root
from src.analysis.group_law import load_population
from src.hardware import resolve_device
from src.repro import environment_record
from src.storage import new_run_dir, write_json
from src.symbolic_f13 import SymbolicFStarMul, classify_pairs, from_fit
from src.tasks.modular import ModularMultiplicationStar

CKPT = (
    ROOT
    / "results"
    / "runs"
    / "20260909_104844_held_row"
    / "complex_h1_held_row"
    / "checkpoints"
    / "successful.pt"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=CKPT)
    args = parser.parse_args()
    device = resolve_device("cuda")
    run_dir = new_run_dir(ROOT, "phase5_symbolic")
    write_json(run_dir / "meta.json", environment_record(str(ROOT)))
    task = ModularMultiplicationStar(p=13)
    a, b, x, y = task.full_domain(device)
    a_np = a.cpu().numpy().astype(np.int64)
    b_np = b.cpu().numpy().astype(np.int64)
    y_np = y.cpu().numpy().astype(np.int64)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = load_population(ckpt, device)
    neural = model(x).argmax(dim=-1).cpu().numpy()  # [B, P]
    dlog = discrete_log_table(13)
    n = 12
    certs = []
    n_sym_eq_y = 0
    n_sym_eq_neural = 0
    n_pair_mismatch_sym_y = 0
    for i in range(model.population):
        xy = ckpt["E"][i, :, 0, :].numpy().astype(np.float64)
        fit = fit_dlog_circle(xy, dlog, n)
        cert = from_fit(13, fit)
        pred = classify_pairs(cert, a_np, b_np, dlog)
        eq_y = int((pred == y_np).sum())
        eq_n = int((pred == neural[:, i]).sum())
        n_pair_mismatch_sym_y += 144 - eq_y
        n_sym_eq_y += int(eq_y == 144)
        n_sym_eq_neural += int(eq_n == 144)
        rec = cert.as_dict()
        rec["index"] = int(ckpt["indices"][i]) if "indices" in ckpt else i
        rec["r2"] = float(fit["r2"])
        rec["n_sym_eq_label"] = eq_y
        rec["n_sym_eq_neural"] = eq_n
        certs.append(rec)
    summary = {
        "n": model.population,
        "params_neural": model.n_params_per_network(),
        "symbolic_degrees_of_freedom": "k in (Z/12Z)* (2 bits among 4 units), conjugate (1 bit), scale, phi",
        "primitive_root": int(primitive_root(13)),
        "n_symbolic_exact_on_labels": n_sym_eq_y,
        "n_symbolic_equals_neural_argmax": n_sym_eq_neural,
        "pair_mismatches_symbolic_vs_label": n_pair_mismatch_sym_y,
        "checkpoint": str(args.checkpoint),
        "program": (
            "theta(a)=2*pi*k*log_g(a)/12  [or minus if conjugate]; "
            "z=s*exp(i(phi+theta)); h=z(a)z(b); "
            "class=argmax_c h·(s^2 R_{2 phi} unit_k(c))"
        ),
    }
    write_json(run_dir / "metrics" / "symbolic.json", {"summary": summary, "certificates": certs})
    print(
        f"symbolic exact on labels {n_sym_eq_y}/{model.population}; "
        f"equals neural argmax {n_sym_eq_neural}/{model.population}; "
        f"pair mismatches vs label {n_pair_mismatch_sym_y}"
    )
    print("wrote", run_dir / "metrics" / "symbolic.json")


if __name__ == "__main__":
    main()
