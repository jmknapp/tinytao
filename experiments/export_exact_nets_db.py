#!/usr/bin/env python3
"""Populate MySQL tinytao.nets from an exact-solver checkpoint.

Credentials: MYSQL_USER, MYSQL_PASSWORD, optional MYSQL_HOST / MYSQL_PORT /
MYSQL_DATABASE (default tinytao).

Each row is one exact net: original population id, torus_crt classification,
and the 390 learned floats (E, W_out, b_out).

  PYTHONPATH=. python experiments/export_exact_nets_db.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.nets_db import DEFAULT_RUN, class_counts, connect, mysql_settings, n_learned, replace_run_from_arrays

DEFAULT_CKPT = (
    ROOT
    / "results"
    / "runs"
    / DEFAULT_RUN
    / "complex_h3_p31_held_row_pop8192"
    / "checkpoints"
    / "successful.pt"
)
P = 31


def load_checkpoint(path: Path):
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    E = ckpt["E"].detach().cpu().numpy()
    W = ckpt["W_out"].detach().cpu().numpy()
    b = ckpt["b_out"].detach().cpu().numpy()
    if "indices" in ckpt:
        nets = ckpt["indices"].detach().cpu().numpy()
    else:
        nets = __import__("numpy").arange(E.shape[0])
    return E, W, b, nets


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--run", default=DEFAULT_RUN)
    ap.add_argument("--p", type=int, default=P)
    args = ap.parse_args()
    cfg = mysql_settings()
    if not args.ckpt.is_file():
        raise SystemExit(
            f"checkpoint not found: {args.ckpt}\n"
            "Pass --ckpt to successful.pt from the p=31 H=3 pop8192 run."
        )
    E, W, b, nets = load_checkpoint(args.ckpt)
    n, heads = int(E.shape[1]), int(E.shape[2])
    want = n_learned(n, heads)
    print(
        f"loading {args.ckpt}  exact={E.shape[0]}  E{tuple(E.shape)}  "
        f"W{tuple(W.shape)}  b{tuple(b.shape)}  n_params={want}",
        flush=True,
    )
    print(
        f"mysql {cfg['user']}@{cfg['host']}:{cfg['port']}/{cfg['database']}.nets",
        flush=True,
    )
    conn = connect()
    counts = replace_run_from_arrays(
        conn, run=args.run, p=args.p, nets=nets, E=E, W_out=W, b_out=b, classify=True
    )
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) AS c FROM nets WHERE run = %s", (args.run,))
        n_rows = int(cur.fetchone()["c"])
    print("rows", n_rows)
    print("classification", class_counts(conn, args.run) or counts)
    conn.close()


if __name__ == "__main__":
    main()
