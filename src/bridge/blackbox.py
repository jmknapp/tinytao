"""Discoverer-facing map: evaluation only. No residue table, no certificate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class BlackBoxMap:
    name: str
    halt: int
    _T: Callable[[int], int]
    _F_odd: Callable[[int], int]

    def T(self, n: int) -> int:
        return self._T(n)

    def F_odd(self, n: int) -> int:
        if n % 2 == 0:
            raise ValueError("F_odd expects an odd integer")
        return self._F_odd(n)

    def F_odd_k(self, n: int, k: int) -> int:
        x = n
        for _ in range(k):
            x = self.F_odd(x)
        return x


_ALLOWED = (
    "trivial_plus_one",
    "easy_half_collatz",
    "medium_five_on_1_mod_16",
    "hard_five_on_1_mod_16",
    "prospective_seven_on_1_mod_32",
    "prospective_seven_and_three_mod_32",
    "prospective_seven_on_1_mod_64",
    "control_five_on_1_mod_4",
)


def get_blackbox(name: str) -> BlackBoxMap:
    if name not in _ALLOWED:
        raise KeyError(name)
    # Import catalog only to construct T. Callers must not use the BridgeMap.
    from src.bridge import catalog as _catalog

    factory = {
        "trivial_plus_one": _catalog.map_trivial_plus_one,
        "easy_half_collatz": _catalog.map_easy_half_collatz,
        "medium_five_on_1_mod_16": _catalog.map_medium_five_on_1_mod_16,
        "hard_five_on_1_mod_16": _catalog.map_hard_five_on_1_mod_16,
        "prospective_seven_on_1_mod_32": _catalog.map_prospective_seven_on_1_mod_32,
        "prospective_seven_and_three_mod_32": _catalog.map_prospective_seven_and_three_mod_32,
        "prospective_seven_on_1_mod_64": _catalog.map_prospective_seven_on_1_mod_64,
        "control_five_on_1_mod_4": _catalog.map_control_scrambled,
    }[name]
    inner = factory()
    return BlackBoxMap(name=inner.name, halt=inner.halt, _T=inner.T, _F_odd=inner.F_odd)
