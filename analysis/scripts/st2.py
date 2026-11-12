"""Exact order-2 Shapley–Taylor terms for four named coalition players.

Bits are P1 P2 P3 P4, left to right; P1 is the most significant bit.
These are raw score differences, not normalized values or Shapley interaction
indices (SII). Higher-order effects are distributed among their contained pairs.
This module has no model, image, table or third-party dependencies.
"""
from __future__ import annotations

from collections.abc import Mapping
from itertools import combinations
import math
from numbers import Real
from typing import TypedDict

PARTS = ("P1", "P2", "P3", "P4")
ALL_BITS = tuple(format(mask, "04b") for mask in range(16))
DEFAULT_ABS_TOL = 1e-10
DEFAULT_REL_TOL = 1e-10


class ST2Result(TypedDict):
    singletons: dict[str, float]
    pairs: dict[str, float]
    baseline: float
    full: float
    efficiency_error: float
    efficiency_passed: bool


def _finite_real(value: Real, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a real number, not bool or a numeric string")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{label} must be finite in float arithmetic") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _sum(values) -> float:
    try:
        result = math.fsum(values)
    except OverflowError as exc:
        raise ValueError("ST2 arithmetic exceeds finite float range") from exc
    if not math.isfinite(result):
        raise ValueError("ST2 arithmetic exceeds finite float range")
    return result


def exact_st2(
    values: Mapping[str, float],
    *,
    abs_tol: float = DEFAULT_ABS_TOL,
    rel_tol: float = DEFAULT_REL_TOL,
) -> ST2Result:
    """Return four singleton and six pair terms from exactly sixteen values.

    Scores must be finite real numbers (negative teaching values are allowed).
    Keys must be the sixteen four-character bit strings; no missing/extra keys,
    numeric keys, or implicit zero filling is allowed. Input order is irrelevant.

    I_i = v({i}) - v(empty). For pair i,j and the remaining players k,l,
    I_ij = D_ij(empty)/2 + [D_ij(k) + D_ij(l) + D_ij(kl)]/6.
    D is the ordinary second finite difference. No random sampling is used.

    The signed efficiency residual is full - baseline - sum(all ten terms).
    It passes when abs(residual) <= abs_tol + rel_tol * scale, where scale is
    max(abs(full - baseline), abs(sum(all ten terms))). These numerical defaults
    are implementation checks, not approval of a formal experiment's tolerances.
    Invalid types raise TypeError; incomplete keys, non-finite values, negative
    tolerances and arithmetic overflow raise ValueError. Returned terms are never
    adjusted to force efficiency to pass.
    """
    absolute = _finite_real(abs_tol, "abs_tol")
    relative = _finite_real(rel_tol, "rel_tol")
    if absolute < 0 or relative < 0:
        raise ValueError("ST2 tolerances must be nonnegative")
    if not isinstance(values, Mapping):
        raise TypeError("values must be a mapping of sixteen coalition bit strings")
    if len(values) != 16 or set(values) != set(ALL_BITS):
        raise ValueError("values must contain exactly the sixteen keys 0000 through 1111")
    v = [_finite_real(values[bits], f"values[{bits!r}]") for bits in ALL_BITS]
    masks = (8, 4, 2, 1)
    singletons = {part: _sum((v[mask], -v[0])) for part, mask in zip(PARTS, masks)}
    pairs = {}
    for i, j in combinations(range(4), 2):
        a, b = masks[i], masks[j]
        k, l = (masks[index] for index in range(4) if index not in (i, j))

        def difference(context: int) -> float:
            return _sum((v[context | a | b], -v[context | a], -v[context | b], v[context]))

        pairs[f"{PARTS[i]}:{PARTS[j]}"] = _sum(
            (difference(0) / 2, difference(k) / 6, difference(l) / 6, difference(k | l) / 6)
        )
    terms = (*singletons.values(), *pairs.values())
    delta = _sum((v[15], -v[0]))
    allocated = _sum(terms)
    error = _sum((v[15], -v[0], *(-term for term in terms)))
    # Dividing by scale avoids overflowing relative * scale for large tolerances.
    scale = max(abs(delta), abs(allocated))
    passed = abs(error) <= absolute or (scale > 0 and (abs(error) - absolute) / scale <= relative)
    return dict(singletons=singletons, pairs=pairs, baseline=v[0], full=v[15],
                efficiency_error=error, efficiency_passed=passed)
