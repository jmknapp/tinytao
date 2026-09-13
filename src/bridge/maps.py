"""Collatz-like maps on positive integers, plus exact affine residue arithmetic.

Even n always map to n/2. Odd n use a residue-dependent accelerated rule
    F(n) = (a n + b) / 2^{v2(a n + b)}
with a, b odd, so a n + b is even.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from typing import Callable


def v2(n: int) -> int:
    if n == 0:
        raise ValueError("v2(0) undefined")
    n = abs(n)
    v = 0
    while n % 2 == 0:
        n //= 2
        v += 1
    return v


def odd_part(n: int) -> int:
    if n == 0:
        return 0
    return abs(n) >> v2(n)


@dataclass(frozen=True)
class Affine:
    """x |-> A x + B on the integer line (used after solving n = M q + r)."""

    A: int
    B: int

    def __call__(self, q: int) -> int:
        return self.A * q + self.B

    def compose(self, inner: "Affine") -> "Affine":
        # self(inner(q)) = A (A_i q + B_i) + B
        return Affine(self.A * inner.A, self.A * inner.B + self.B)


@dataclass(frozen=True)
class ResidueRule:
    """Rule used when n ≡ r (mod 2^m), n odd: apply (a n + b) then strip 2s."""

    r: int
    m: int
    a: int
    b: int

    def apply(self, n: int) -> int:
        return odd_part(self.a * n + self.b)


@dataclass(frozen=True)
class BridgeMap:
    """Full map T on Z>0. Evens halve; odds use the first matching residue rule.

    Rules must cover every odd residue modulo 2^{max m}.
    """

    name: str
    rules: tuple[ResidueRule, ...]
    halt: int = 1

    @property
    def modulus(self) -> int:
        return 1 << max(rule.m for rule in self.rules)

    def rule_for(self, n: int) -> ResidueRule:
        if n % 2 == 0:
            raise ValueError("even n has no odd-rule")
        for rule in self.rules:
            if n % (1 << rule.m) == rule.r:
                return rule
        raise KeyError(f"no rule for n={n} in {self.name}")

    def T(self, n: int) -> int:
        if n < 1:
            raise ValueError("domain is positive integers")
        if n % 2 == 0:
            return n // 2
        return self.rule_for(n).apply(n)

    def F_odd(self, n: int) -> int:
        """Odd-to-odd accelerated step (assumes n odd)."""
        return self.rule_for(n).apply(n)

    def trajectory(self, n: int, max_steps: int = 10_000) -> list[int]:
        seen_list: list[int] = []
        seen_set: set[int] = set()
        x = n
        for _ in range(max_steps):
            seen_list.append(x)
            if x == self.halt:
                return seen_list
            if x in seen_set:
                seen_list.append(x)
                return seen_list
            seen_set.add(x)
            x = self.T(x)
        raise RuntimeError(f"{self.name}: no halt/cycle from {n} in {max_steps} steps")


def affine_odd_step(rule: ResidueRule, M: int, r: int) -> Affine:
    """Exact F(M q + r) as affine in q, once v2(ar+b) < v2(M)."""
    if M % 2 != 0 or (M & (M - 1)) != 0:
        raise ValueError("M must be a power of 2")
    m_M = v2(M)
    if r % 2 == 0:
        raise ValueError("r must be odd")
    lin = rule.a * r + rule.b
    v = v2(lin)
    if v >= m_M:
        raise ValueError(
            f"v2({rule.a}*{r}+{rule.b})={v} >= v2(M)={m_M}; lift the modulus"
        )
    A = rule.a * (M >> v)
    B = lin >> v
    if B % 2 == 0:
        raise ValueError(f"expected odd image, got B={B}")
    return Affine(A, B)


def covering_rules(m: int, choose: Callable[[int], tuple[int, int]]) -> tuple[ResidueRule, ...]:
    """One rule per odd residue mod 2^m."""
    M = 1 << m
    rules = []
    for r in range(1, M, 2):
        a, b = choose(r)
        rules.append(ResidueRule(r=r, m=m, a=a, b=b))
    return tuple(rules)
