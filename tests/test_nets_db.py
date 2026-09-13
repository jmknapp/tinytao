"""SQLite nets table: 390 learned parameters plus torus_crt classification."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.nets_db import (
    class_counts,
    connect,
    fetch_net,
    n_learned,
    pack_params,
    replace_run_from_arrays,
    unpack_params,
)
from src.nets_db import _load_torus_crt

_mod = _load_torus_crt()
apply_scale_rot = _mod.apply_scale_rot
certify_torus_net = _mod.certify_torus_net
discrete_log_table = _mod.discrete_log_table
exact_unit_circle = _mod.exact_unit_circle


def _char_head(p: int, k: int, scale: float = 1.0, phi: float = 0.0) -> np.ndarray:
    n = p - 1
    dlog = discrete_log_table(p)
    unit = exact_unit_circle(k, dlog, n, False)
    return apply_scale_rot(unit, scale, phi)


def test_pack_roundtrip_390():
    n, heads = 30, 3
    rng = np.random.default_rng(0)
    e = rng.normal(size=(n, heads, 2)).astype(np.float32)
    w = rng.normal(size=(2 * heads, n)).astype(np.float32)
    b = rng.normal(size=(n,)).astype(np.float32)
    params = pack_params(e, w, b)
    assert params.shape == (390,)
    assert n_learned(n, heads) == 390
    got = unpack_params(params, n, heads)
    np.testing.assert_array_equal(got["E"], e)
    np.testing.assert_array_equal(got["W_out"], w)
    np.testing.assert_array_equal(got["b_out"], b)


def test_populate_and_classify(tmp_path: Path):
    p, n, heads = 13, 12, 3
    live = _char_head(p, k=1, scale=1.0)
    dead = np.zeros((n, 2), dtype=np.float64)
    e_f = np.stack([live, dead, dead + 1e-9], axis=1)
    # Three order-2 characters: lcm=2 ≠ 12 → unclassified.
    even = _char_head(p, k=2, scale=1.2)
    e_u = np.stack([even, even * 0.8, even * 1.1], axis=1)
    E = np.stack([e_f, e_u], axis=0).astype(np.float32)
    W = np.zeros((2, 2 * heads, n), dtype=np.float32)
    b = np.zeros((2, n), dtype=np.float32)
    W[0, 0, :] = 1.0
    db = tmp_path / "nets.sqlite"
    conn = connect(db)
    counts = replace_run_from_arrays(
        conn, run="unit", p=p, nets=np.array([7, 43]), E=E, W_out=W, b_out=b
    )
    assert certify_torus_net(e_f, p)["label"] == "FAMILY_F"
    assert counts["FAMILY_F"] == 1
    assert counts["EXACT-UNCLASSIFIED"] == 1
    rec = fetch_net(conn, "unit", 7)
    assert rec is not None
    assert rec["classification"] == "FAMILY_F"
    assert rec["params"].shape == (n_learned(n, heads),)
    np.testing.assert_allclose(rec["E"], e_f, atol=1e-6)
    u = fetch_net(conn, "unit", 43)
    assert u is not None
    assert u["classification"] == "EXACT-UNCLASSIFIED"
    assert class_counts(conn, "unit") == counts
    n_params = conn.execute("SELECT n_params FROM nets WHERE net = 7").fetchone()[0]
    assert int(n_params) == n_learned(n, heads)
    conn.close()
    # reopen
    conn = sqlite3.connect(str(db))
    n_rows = conn.execute("SELECT COUNT(*) FROM nets").fetchone()[0]
    assert n_rows == 2
    conn.close()
