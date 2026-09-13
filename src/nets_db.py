"""SQLite table of exact Model C nets: 390 learned parameters plus classification.

Parameter order (length 390 for p=31, H=3):
    E[n, H, 2] row-major, then W_out[2H, n] row-major, then b_out[n].
    n = p-1, 2H = 6.  30*3*2 + 6*30 + 30 = 390.
"""

from __future__ import annotations

import importlib.util
import sqlite3
from pathlib import Path
from typing import Any, Iterator

import numpy as np


def _load_torus_crt():
    """Load torus_crt without importing src.analysis.__init__ (torch/tqdm)."""
    global _TORUS
    if _TORUS is not None:
        return _TORUS
    path = Path(__file__).resolve().parent / "analysis" / "torus_crt.py"
    spec = importlib.util.spec_from_file_location("_tinytao_torus_crt", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _TORUS = mod
    return mod


_TORUS = None


def _certify_torus_net(xy, p):
    return _load_torus_crt().certify_torus_net(xy, p)

SCHEMA = """
CREATE TABLE IF NOT EXISTS nets (
    id INTEGER PRIMARY KEY,
    run TEXT NOT NULL,
    net INTEGER NOT NULL,
    slice INTEGER NOT NULL,
    classification TEXT NOT NULL,
    p INTEGER NOT NULL,
    heads INTEGER NOT NULL,
    n INTEGER NOT NULL,
    n_params INTEGER NOT NULL,
    lcm INTEGER,
    break_at TEXT,
    params BLOB NOT NULL,
    UNIQUE(run, net)
);
CREATE INDEX IF NOT EXISTS idx_nets_class ON nets(classification);
CREATE INDEX IF NOT EXISTS idx_nets_run_class ON nets(run, classification);
"""

DEFAULT_RUN = "20260913_133152_p31_h3_three_torus_pop8192"


def n_learned(n: int, heads: int) -> int:
    return n * heads * 2 + (2 * heads) * n + n


def pack_params(e: np.ndarray, w_out: np.ndarray, b_out: np.ndarray) -> np.ndarray:
    """Flatten E, W_out, b_out into one float32 vector."""
    e = np.asarray(e, dtype=np.float32)
    w_out = np.asarray(w_out, dtype=np.float32)
    b_out = np.asarray(b_out, dtype=np.float32)
    if e.ndim != 3 or e.shape[-1] != 2:
        raise ValueError(f"E expected [n, H, 2], got {e.shape}")
    n, heads, _ = e.shape
    if w_out.shape != (2 * heads, n):
        raise ValueError(f"W_out expected [{2 * heads}, {n}], got {w_out.shape}")
    if b_out.shape != (n,):
        raise ValueError(f"b_out expected [{n}], got {b_out.shape}")
    return np.concatenate([e.reshape(-1), w_out.reshape(-1), b_out.reshape(-1)])


def unpack_params(params: np.ndarray, n: int, heads: int) -> dict[str, np.ndarray]:
    params = np.asarray(params, dtype=np.float32).reshape(-1)
    want = n_learned(n, heads)
    if params.size != want:
        raise ValueError(f"params length {params.size} != {want}")
    n_e = n * heads * 2
    n_w = 2 * heads * n
    e = params[:n_e].reshape(n, heads, 2)
    w_out = params[n_e : n_e + n_w].reshape(2 * heads, n)
    b_out = params[n_e + n_w :]
    return {"E": e, "W_out": w_out, "b_out": b_out}


def connect(path: str | Path) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def insert_net(
    conn: sqlite3.Connection,
    *,
    run: str,
    net: int,
    slice_: int,
    classification: str,
    p: int,
    e: np.ndarray,
    w_out: np.ndarray,
    b_out: np.ndarray,
    lcm: int | None = None,
    break_at: str | None = None,
) -> None:
    e = np.asarray(e)
    n, heads, _ = e.shape
    params = pack_params(e, w_out, b_out)
    conn.execute(
        """
        INSERT INTO nets (
            run, net, slice, classification, p, heads, n, n_params, lcm, break_at, params
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run,
            int(net),
            int(slice_),
            str(classification),
            int(p),
            int(heads),
            int(n),
            int(params.size),
            None if lcm is None else int(lcm),
            break_at,
            params.tobytes(),
        ),
    )


def row_to_record(row: sqlite3.Row) -> dict[str, Any]:
    n, heads = int(row["n"]), int(row["heads"])
    params = np.frombuffer(row["params"], dtype=np.float32).copy()
    tensors = unpack_params(params, n, heads)
    return {
        "run": row["run"],
        "net": int(row["net"]),
        "slice": int(row["slice"]),
        "classification": row["classification"],
        "p": int(row["p"]),
        "heads": heads,
        "n": n,
        "n_params": int(row["n_params"]),
        "lcm": row["lcm"],
        "break_at": row["break_at"],
        "params": params,
        **tensors,
    }


def fetch_net(conn: sqlite3.Connection, run: str, net: int) -> dict[str, Any] | None:
    cur = conn.execute("SELECT * FROM nets WHERE run = ? AND net = ?", (run, int(net)))
    row = cur.fetchone()
    return None if row is None else row_to_record(row)


def iter_nets(conn: sqlite3.Connection, run: str | None = None) -> Iterator[dict[str, Any]]:
    if run is None:
        cur = conn.execute("SELECT * FROM nets ORDER BY run, net")
    else:
        cur = conn.execute("SELECT * FROM nets WHERE run = ? ORDER BY net", (run,))
    for row in cur:
        yield row_to_record(row)


def class_counts(conn: sqlite3.Connection, run: str | None = None) -> dict[str, int]:
    if run is None:
        cur = conn.execute(
            "SELECT classification, COUNT(*) AS c FROM nets GROUP BY classification ORDER BY classification"
        )
    else:
        cur = conn.execute(
            "SELECT classification, COUNT(*) AS c FROM nets WHERE run = ? GROUP BY classification ORDER BY classification",
            (run,),
        )
    return {str(r["classification"]): int(r["c"]) for r in cur}


def replace_run_from_arrays(
    conn: sqlite3.Connection,
    *,
    run: str,
    p: int,
    nets: np.ndarray,
    E: np.ndarray,
    W_out: np.ndarray,
    b_out: np.ndarray,
    classify: bool = True,
    labels: list[str] | None = None,
) -> dict[str, int]:
    """Wipe one run and insert every exact net. E [P, n, H, 2]."""
    E = np.asarray(E)
    W_out = np.asarray(W_out)
    b_out = np.asarray(b_out)
    nets = np.asarray(nets, dtype=np.int64)
    pop = int(E.shape[0])
    if W_out.shape[0] != pop or b_out.shape[0] != pop or nets.shape[0] != pop:
        raise ValueError("E, W_out, b_out, nets must share leading dimension")
    conn.execute("DELETE FROM nets WHERE run = ?", (run,))
    counts: dict[str, int] = {}
    for i in range(pop):
        if labels is not None:
            label = str(labels[i])
            lcm = None
            break_at = None
        elif classify:
            cert = _certify_torus_net(E[i], p)
            label = str(cert["label"])
            lcm = int(cert["lcm"])
            break_at = cert.get("break_at")
        else:
            label = "UNKNOWN"
            lcm = None
            break_at = None
        insert_net(
            conn,
            run=run,
            net=int(nets[i]),
            slice_=i,
            classification=label,
            p=p,
            e=E[i],
            w_out=W_out[i],
            b_out=b_out[i],
            lcm=lcm,
            break_at=None if break_at is None else str(break_at),
        )
        counts[label] = counts.get(label, 0) + 1
    conn.commit()
    return counts
