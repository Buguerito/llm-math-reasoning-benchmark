import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from math_benchmark.validators.models import ValidationResult


@dataclass(frozen=True)
class Interval:
    left: Decimal
    left_closed: bool
    right: Decimal
    right_closed: bool


def _endpoint(text: str) -> Decimal:
    normalized = text.strip().lower().replace("∞", "inf").replace("infinity", "inf")
    if normalized in {"inf", "+inf"}:
        return Decimal("Infinity")
    if normalized == "-inf":
        return Decimal("-Infinity")
    value = Decimal(normalized)
    if not value.is_finite():
        raise ValueError("invalid endpoint")
    return value


def _parse_component(text: str) -> Interval:
    match = re.fullmatch(r"\s*([\[(])\s*([^,]+)\s*,\s*([^,]+)\s*([\])])\s*", text)
    if not match:
        raise ValueError("malformed interval notation")
    left, right = _endpoint(match.group(2)), _endpoint(match.group(3))
    left_closed, right_closed = match.group(1) == "[", match.group(4) == "]"
    if left > right or (left == right and not (left_closed and right_closed)):
        raise ValueError("invalid interval bounds")
    if (left.is_infinite() and left_closed) or (right.is_infinite() and right_closed):
        raise ValueError("infinite endpoints must be open")
    return Interval(left, left_closed, right, right_closed)


def _merge(intervals: list[Interval]) -> tuple[Interval, ...]:
    ordered = sorted(intervals, key=lambda item: (item.left, not item.left_closed, item.right))
    merged: list[Interval] = []
    for current in ordered:
        if not merged:
            merged.append(current)
            continue
        previous = merged[-1]
        touches = current.left == previous.right and (current.left_closed or previous.right_closed)
        if current.left < previous.right or touches:
            if current.right > previous.right:
                right, right_closed = current.right, current.right_closed
            elif current.right < previous.right:
                right, right_closed = previous.right, previous.right_closed
            else:
                right, right_closed = previous.right, previous.right_closed or current.right_closed
            merged[-1] = Interval(previous.left, previous.left_closed, right, right_closed)
        else:
            merged.append(current)
    return tuple(merged)


def _parse(value: str) -> tuple[Interval, ...]:
    stripped = value.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        body = stripped[1:-1].strip()
        if not body:
            return ()
        try:
            points = sorted({_endpoint(part) for part in body.split(",")})
        except InvalidOperation as exc:
            raise ValueError("malformed finite set") from exc
        if any(point.is_infinite() for point in points):
            raise ValueError("finite sets cannot contain infinity")
        return tuple(Interval(point, True, point, True) for point in points)

    normalized = re.sub(r"\s+(?:union|u)\s+|\s*∪\s*", "|", stripped, flags=re.IGNORECASE)
    parts = normalized.split("|")
    if any(not part.strip() for part in parts):
        raise ValueError("malformed interval union")
    try:
        return _merge([_parse_component(part) for part in parts])
    except InvalidOperation as exc:
        raise ValueError("malformed interval endpoint") from exc


def set_interval_equivalent(actual: str, expected: str) -> ValidationResult:
    try:
        actual_parsed = _parse(actual)
        expected_parsed = _parse(expected)
    except ValueError as exc:
        return ValidationResult(None, True, actual.strip(), str(exc))
    return ValidationResult(actual_parsed == expected_parsed, False, repr(actual_parsed), "set equivalence")
