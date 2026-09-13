"""Unit tests for T^m CRT / torus certification (numpy only, no training)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.torus_crt import (  # noqa: E402
    apply_scale_rot,
    certify_torus_net,
    discrete_log_table,
    exact_unit_circle,
)


def _char_head(p: int, k: int, scale: float = 1.0, phi: float = 0.0, conjugate: bool = False) -> np.ndarray:
    n = p - 1
    dlog = discrete_log_table(p)
    unit = exact_unit_circle(k, dlog, n, conjugate)
    return apply_scale_rot(unit, scale, phi)


def test_synthetic_t1_overcomplete_h3():
    """One faithful circle + two near-zero heads => T^1 / FAMILY_F, not T^3."""
    p = 13
    n = p - 1
    live = _char_head(p, k=1, scale=1.0)
    dead = np.zeros((n, 2), dtype=np.float64)
    xy = np.stack([live, dead, dead * 0.0 + 1e-9], axis=1)
    cert = certify_torus_net(xy, p)
    assert cert["decoder_exact"], cert
    assert cert["ablation_ok"], cert
    assert cert["certified"], cert
    assert cert["m"] == 1, cert
    assert cert["lcm"] == n, cert
    assert cert["label"] == "FAMILY_F", cert


def test_synthetic_t3_p31_orders_2_3_5():
    """Heads with image orders 2,3,5 (gcd in {15,10,6}) => T^3; ablation load-bearing."""
    p = 31
    n = p - 1
    assert math.gcd(15, 30) == 15 and n // 15 == 2
    assert math.gcd(10, 30) == 10 and n // 10 == 3
    assert math.gcd(6, 30) == 6 and n // 6 == 5
    h2 = _char_head(p, k=15, scale=1.0)
    h3 = _char_head(p, k=10, scale=1.0)
    h5 = _char_head(p, k=6, scale=1.0)
    xy = np.stack([h2, h3, h5], axis=1)
    cert = certify_torus_net(xy, p)
    assert cert["m"] == 3, cert
    assert cert["lcm"] == n, cert
    assert cert["decoder_exact"], cert
    assert cert["ablation_ok"], cert
    assert cert["certified"], cert
    assert cert["label"] == "TORUS_T3", cert
    for row in cert["ablation"]:
        assert row["exact_after"] is False, row


def test_three_identical_faithful_heads_are_t1():
    """Three copies of the same faithful winding => m=1, not m=3."""
    p = 13
    n = p - 1
    h = _char_head(p, k=1, scale=1.0)
    xy = np.stack([h, h.copy(), h.copy()], axis=1)
    cert = certify_torus_net(xy, p)
    assert cert["certified"], cert
    assert cert["m"] == 1, cert
    assert cert["lcm"] == n, cert
    assert cert["label"] == "FAMILY_F", cert
    assert len(cert["live_heads"]) == 1, cert


def main() -> None:
    test_synthetic_t1_overcomplete_h3()
    print("ok: T^1 overcomplete H=3")
    test_synthetic_t3_p31_orders_2_3_5()
    print("ok: T^3 p=31 orders 2,3,5")
    test_three_identical_faithful_heads_are_t1()
    print("ok: identical faithful heads -> T^1")
    print("ALL PASSED")


if __name__ == "__main__":
    main()
