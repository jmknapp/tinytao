"""Hidden calibration maps for the Collatz bridge. Discoverer must not import this."""

from __future__ import annotations

from src.bridge.maps import BridgeMap, ResidueRule, covering_rules


def _const(a: int, b: int):
    return lambda r: (a, b)


def map_trivial_plus_one() -> BridgeMap:
    """F(n)=(n+1)/2^{v2(n+1)} on every odd n. One-step descent. Too easy."""
    return BridgeMap(
        name="trivial_plus_one",
        rules=covering_rules(1, _const(1, 1)),
    )


def map_easy_half_collatz() -> BridgeMap:
    """3n+1 only on n≡1 (mod 4); n≡3 (mod 4) uses n+1.

    Accelerated odd map still one-step descends (certificate uses mod 8).
    Raw 3n+1 branch increases before stripping 2s, so it is Collatz-like
    to the eye but not open.
    """

    def choose(r: int) -> tuple[int, int]:
        if r % 4 == 1:
            return 3, 1
        return 1, 1

    return BridgeMap(name="easy_half_collatz", rules=covering_rules(2, choose))


def map_medium_five_on_1_mod_16() -> BridgeMap:
    """5n+1 only when n≡1 (mod 16); other odd classes use n+1.

    Thin expanding class. Two residues mod 32 need a k=2 lemma; the rest
    one-step descend. Still a finite 2-adic covering.
    """

    def choose(r: int) -> tuple[int, int]:
        if r % 16 == 1:
            return 5, 1
        return 1, 1

    return BridgeMap(name="medium_five_on_1_mod_16", rules=covering_rules(4, choose))


def map_hard_five_on_1_mod_16() -> BridgeMap:
    """5n+1 on n≡1 (mod 16); 3n+1 on n≡9 (mod 16); n+1 elsewhere.

    Stronger expansion on a thin class. Still expected to have a finite
    2-adic covering certificate, unlike 5x+1 on all odds.
    """

    def choose(r: int) -> tuple[int, int]:
        if r % 16 == 1:
            return 5, 1
        if r % 16 == 9:
            return 3, 1
        return 1, 1

    return BridgeMap(name="hard_five_on_1_mod_16", rules=covering_rules(4, choose))


def map_prospective_seven_on_1_mod_32() -> BridgeMap:
    """7n+1 only when n≡1 (mod 32); other odd classes use n+1.

    Thinner expanding class than the mod-16 maps, different multiplier.
    Hand analysis: M=16 mixes two laws on residue 1; mixed covering at
    M=32 with k=1 on the expanding class (F=28q+1). Preregistered in
    tiny_tao_results/bridge_preregistration_seven_on_1_mod_32.json.
    """

    def choose(r: int) -> tuple[int, int]:
        if r % 32 == 1:
            return 7, 1
        return 1, 1

    return BridgeMap(name="prospective_seven_on_1_mod_32", rules=covering_rules(5, choose))


def map_prospective_seven_and_three_mod_32() -> BridgeMap:
    """7n+1 on n≡1 (mod 32); 3n+1 on n≡17 (mod 32); n+1 elsewhere.

    Thicker expanding set than the 1/32 miss (density 1/16 of odds).
    M=16 residue 1 mixes two expanding laws. Preregistered in
    tiny_tao_results/bridge_preregistration_seven_and_three_mod_32.json.
    """

    def choose(r: int) -> tuple[int, int]:
        if r % 32 == 1:
            return 7, 1
        if r % 32 == 17:
            return 3, 1
        return 1, 1

    return BridgeMap(name="prospective_seven_and_three_mod_32", rules=covering_rules(5, choose))


def map_prospective_seven_on_1_mod_64() -> BridgeMap:
    """7n+1 only when n≡1 (mod 64); other odd classes use n+1.

    Covering lives at M=64 (F=56q+1 on residue 1). The locked neural
    feature bank stops at 32, so 15.6-style sweep should fail with
    residue 1 uncovered. Not a reason to add 64 after the miss.
    Preregistered in
    tiny_tao_results/bridge_preregistration_seven_on_1_mod_64.json.
    """

    def choose(r: int) -> tuple[int, int]:
        if r % 64 == 1:
            return 7, 1
        return 1, 1

    return BridgeMap(name="prospective_seven_on_1_mod_64", rules=covering_rules(6, choose))


def map_control_scrambled() -> BridgeMap:
    """Matched-looking branches with expanding 5n+1 on a class that does
    not admit a small 2-adic covering. Used later as a negative control.
    Not claimed terminating; oracle must record that the proof search fails.
    """

    def choose(r: int) -> tuple[int, int]:
        if r % 4 == 1:
            return 5, 1
        return 1, 1

    return BridgeMap(name="control_five_on_1_mod_4", rules=covering_rules(2, choose))


CANDIDATE_MAPS = (
    map_trivial_plus_one,
    map_easy_half_collatz,
    map_medium_five_on_1_mod_16,
    map_hard_five_on_1_mod_16,
    map_prospective_seven_on_1_mod_32,
    map_prospective_seven_and_three_mod_32,
    map_prospective_seven_on_1_mod_64,
    map_control_scrambled,
)


_FACTORY = {
    "trivial_plus_one": map_trivial_plus_one,
    "easy_half_collatz": map_easy_half_collatz,
    "medium_five_on_1_mod_16": map_medium_five_on_1_mod_16,
    "hard_five_on_1_mod_16": map_hard_five_on_1_mod_16,
    "prospective_seven_on_1_mod_32": map_prospective_seven_on_1_mod_32,
    "prospective_seven_and_three_mod_32": map_prospective_seven_and_three_mod_32,
    "prospective_seven_on_1_mod_64": map_prospective_seven_on_1_mod_64,
    "control_five_on_1_mod_4": map_control_scrambled,
}


def get_bridge_map(name: str) -> BridgeMap:
    """Oracle/verifier only. Discoverer must not call this."""
    if name not in _FACTORY:
        raise KeyError(name)
    return _FACTORY[name]()
