"""Exact rational arithmetic.

No floating point is ever produced by this package.  Values are accepted as
ints, ``fractions.Fraction``, or one of these string forms:

* ``"3"``, ``"-3"``, ``"+3"``
* ``"3/4"``, ``"-3/4"``
* ``"0.125"``, ``"-2.5"`` (a finite decimal literal, converted *exactly*)
* ``" 3 / 4 "`` (surrounding and inner whitespace around ``/`` is ignored)

Anything else -- in particular exponents, ``nan``/``inf``, hexadecimal or
non-finite spellings -- is rejected rather than guessed at.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from fractions import Fraction
import re
from typing import Union

RationalLike = Union[int, str, Fraction]

_INT_RE = re.compile(r"^[+-]?\d+$")
_FRACTION_RE = re.compile(r"^([+-]?\d+)\s*/\s*([+-]?\d+)$")
_DECIMAL_RE = re.compile(r"^[+-]?(?:\d+\.\d*|\.\d+|\d+)$")

__all__ = ["RationalLike", "as_fraction", "to_str", "is_zero", "cmp_key", "RationalError"]


class RationalError(ValueError):
    """Raised when a value cannot be interpreted exactly."""


def as_fraction(value: RationalLike) -> Fraction:
    """Convert ``value`` to an exact :class:`~fractions.Fraction`.

    ``bool`` is deliberately rejected: ``True`` is not the number 1 in a
    contract payload, and silently accepting it would hide schema bugs.
    """
    if isinstance(value, Fraction):
        return value
    if isinstance(value, bool):
        raise RationalError("boolean is not a rational value")
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise RationalError("empty rational literal")
        if _INT_RE.match(text):
            return Fraction(int(text))
        match = _FRACTION_RE.match(text)
        if match:
            numerator, denominator = int(match.group(1)), int(match.group(2))
            if denominator == 0:
                raise RationalError("rational literal has zero denominator")
            return Fraction(numerator, denominator)
        if _DECIMAL_RE.match(text):
            try:
                return Fraction(Decimal(text))
            except (InvalidOperation, ValueError) as exc:  # pragma: no cover
                raise RationalError(f"invalid decimal literal: {value!r}") from exc
        raise RationalError(f"not an exact rational literal: {value!r}")
    raise RationalError(f"unsupported rational type: {type(value).__name__}")


def to_str(value: RationalLike) -> str:
    """Canonical exact string form: ``n`` or ``n/d`` with ``d > 1``, reduced."""
    frac = as_fraction(value)
    if frac.denominator == 1:
        return str(frac.numerator)
    return f"{frac.numerator}/{frac.denominator}"


def is_zero(value: RationalLike) -> bool:
    return as_fraction(value) == 0


def cmp_key(value: RationalLike) -> Fraction:
    """Exact comparison key for sorting (never a float)."""
    return as_fraction(value)
