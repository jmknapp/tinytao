"""MySQL tinytao.nets: 390 learned parameters plus torus_crt classification."""

from __future__ import annotations

import os
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
    mysql_settings,
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

TEST_RUN = "__tinytao_nets_db_test__"


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


def test_mysql_settings_requires_user():
    old = os.environ.pop("MYSQL_USER", None)
    old_u = os.environ.pop("MYSQL_USERNAME", None)
    try:
        try:
            mysql_settings()
            raise AssertionError("expected RuntimeError")
        except RuntimeError as exc:
            assert "MYSQL_USER" in str(exc)
    finally:
        if old is not None:
            os.environ["MYSQL_USER"] = old
        if old_u is not None:
            os.environ["MYSQL_USERNAME"] = old_u


def test_populate_and_classify():
    if not os.environ.get("MYSQL_USER") and not os.environ.get("MYSQL_USERNAME"):
        print("skip test_populate_and_classify: MYSQL_USER unset")
        return
    p, n, heads = 13, 12, 3
    live = _char_head(p, k=1, scale=1.0)
    dead = np.zeros((n, 2), dtype=np.float64)
    e_f = np.stack([live, dead, dead + 1e-9], axis=1)
    even = _char_head(p, k=2, scale=1.2)
    e_u = np.stack([even, even * 0.8, even * 1.1], axis=1)
    E = np.stack([e_f, e_u], axis=0).astype(np.float32)
    W = np.zeros((2, 2 * heads, n), dtype=np.float32)
    b = np.zeros((2, n), dtype=np.float32)
    W[0, 0, :] = 1.0
    conn = connect()
    try:
        counts = replace_run_from_arrays(
            conn, run=TEST_RUN, p=p, nets=np.array([7, 43]), E=E, W_out=W, b_out=b
        )
        assert certify_torus_net(e_f, p)["label"] == "FAMILY_F"
        assert counts["FAMILY_F"] == 1
        assert counts["EXACT-UNCLASSIFIED"] == 1
        rec = fetch_net(conn, TEST_RUN, 7)
        assert rec is not None
        assert rec["classification"] == "FAMILY_F"
        assert rec["params"].shape == (n_learned(n, heads),)
        np.testing.assert_allclose(rec["E"], e_f, atol=1e-6)
        u = fetch_net(conn, TEST_RUN, 43)
        assert u is not None
        assert u["classification"] == "EXACT-UNCLASSIFIED"
        assert class_counts(conn, TEST_RUN) == counts
        with conn.cursor() as cur:
            cur.execute("SELECT n_params FROM nets WHERE run = %s AND net = 7", (TEST_RUN,))
            n_params = int(cur.fetchone()["n_params"])
        assert n_params == n_learned(n, heads)
    finally:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM nets WHERE run = %s", (TEST_RUN,))
        conn.commit()
        conn.close()
