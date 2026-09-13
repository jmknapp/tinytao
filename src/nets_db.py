"""MySQL table `tinytao.nets`: 390 learned parameters plus classification.

Credentials from the environment (never hardcoded):

    MYSQL_HOST       default 127.0.0.1
    MYSQL_PORT       default 3306
    MYSQL_USER       required
    MYSQL_PASSWORD   or MYSQL_PWD
    MYSQL_DATABASE   default tinytao

Parameter order (length 390 for p=31, H=3):
    E[n, H, 2] row-major, then W_out[2H, n] row-major, then b_out[n].
    n = p-1, 2H = 6.  30*3*2 + 6*30 + 30 = 390.
"""

from __future__ import annotations

import importlib.util
import os
import re
from pathlib import Path
from typing import Any, Iterator

import numpy as np

DEFAULT_RUN = "20260913_133152_p31_h3_three_torus_pop8192"
DEFAULT_DATABASE = "tinytao"
TABLE = "nets"

SCHEMA = """
CREATE TABLE IF NOT EXISTS nets (
    id INT NOT NULL AUTO_INCREMENT,
    run VARCHAR(128) NOT NULL,
    net INT NOT NULL,
    slice_idx INT NOT NULL,
    classification VARCHAR(64) NOT NULL,
    p INT NOT NULL,
    heads INT NOT NULL,
    n INT NOT NULL,
    n_params INT NOT NULL,
    lcm INT NULL,
    break_at VARCHAR(64) NULL,
    params MEDIUMBLOB NOT NULL,
    PRIMARY KEY (id),
    UNIQUE KEY uq_run_net (run, net),
    KEY idx_class (classification),
    KEY idx_run_class (run, classification)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""

_TORUS = None


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


def _certify_torus_net(xy, p):
    return _load_torus_crt().certify_torus_net(xy, p)


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


def mysql_settings() -> dict[str, Any]:
    user = os.environ.get("MYSQL_USER") or os.environ.get("MYSQL_USERNAME")
    if not user:
        raise RuntimeError(
            "MySQL user missing. Set MYSQL_USER and MYSQL_PASSWORD "
            "(host MYSQL_HOST, port MYSQL_PORT, database MYSQL_DATABASE, default tinytao)."
        )
    database = os.environ.get("MYSQL_DATABASE", DEFAULT_DATABASE)
    if not re.fullmatch(r"[A-Za-z0-9_]+", database):
        raise ValueError(f"invalid MYSQL_DATABASE {database!r}")
    port_raw = os.environ.get("MYSQL_PORT", "3306")
    try:
        port = int(port_raw)
    except ValueError as exc:
        raise ValueError(f"invalid MYSQL_PORT {port_raw!r}") from exc
    return {
        "host": os.environ.get("MYSQL_HOST", "127.0.0.1"),
        "port": port,
        "user": user,
        "password": os.environ.get("MYSQL_PASSWORD") or os.environ.get("MYSQL_PWD") or "",
        "database": database,
        "unix_socket": os.environ.get("MYSQL_UNIX_SOCKET") or None,
    }


def connect():
    """Connect, create database `tinytao` if needed, ensure table `nets`."""
    import pymysql
    from pymysql.cursors import DictCursor

    cfg = mysql_settings()
    kwargs: dict[str, Any] = {
        "host": cfg["host"],
        "port": cfg["port"],
        "user": cfg["user"],
        "password": cfg["password"],
        "charset": "utf8mb4",
        "autocommit": False,
        "cursorclass": DictCursor,
    }
    if cfg["unix_socket"]:
        kwargs["unix_socket"] = cfg["unix_socket"]
        kwargs.pop("host", None)
        kwargs.pop("port", None)
    conn = pymysql.connect(**kwargs)
    db = cfg["database"]
    with conn.cursor() as cur:
        cur.execute(f"CREATE DATABASE IF NOT EXISTS `{db}`")
        cur.execute(f"USE `{db}`")
        cur.execute(SCHEMA)
    conn.commit()
    conn.select_db(db)
    return conn


def insert_net(
    conn,
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
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO nets (
                run, net, slice_idx, classification, p, heads, n, n_params, lcm, break_at, params
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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


def row_to_record(row: dict[str, Any]) -> dict[str, Any]:
    n, heads = int(row["n"]), int(row["heads"])
    raw = row["params"]
    params = np.frombuffer(bytes(raw), dtype=np.float32).copy()
    tensors = unpack_params(params, n, heads)
    return {
        "run": row["run"],
        "net": int(row["net"]),
        "slice": int(row["slice_idx"]),
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


def fetch_net(conn, run: str, net: int) -> dict[str, Any] | None:
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM nets WHERE run = %s AND net = %s", (run, int(net)))
        row = cur.fetchone()
    return None if row is None else row_to_record(row)


def iter_nets(conn, run: str | None = None) -> Iterator[dict[str, Any]]:
    with conn.cursor() as cur:
        if run is None:
            cur.execute("SELECT * FROM nets ORDER BY run, net")
        else:
            cur.execute("SELECT * FROM nets WHERE run = %s ORDER BY net", (run,))
        rows = cur.fetchall()
    for row in rows:
        yield row_to_record(row)


def class_counts(conn, run: str | None = None) -> dict[str, int]:
    with conn.cursor() as cur:
        if run is None:
            cur.execute(
                "SELECT classification, COUNT(*) AS c FROM nets GROUP BY classification ORDER BY classification"
            )
        else:
            cur.execute(
                "SELECT classification, COUNT(*) AS c FROM nets WHERE run = %s GROUP BY classification ORDER BY classification",
                (run,),
            )
        rows = cur.fetchall()
    return {str(r["classification"]): int(r["c"]) for r in rows}


def replace_run_from_arrays(
    conn,
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
    with conn.cursor() as cur:
        cur.execute("DELETE FROM nets WHERE run = %s", (run,))
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
