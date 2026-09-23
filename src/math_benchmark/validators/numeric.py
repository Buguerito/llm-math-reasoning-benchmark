from decimal import Decimal, InvalidOperation


def parse_decimal(value: str) -> Decimal:
    """Parse a finite base-10 number and reject NaN or infinities."""
    try:
        parsed = Decimal(value.strip().replace("−", "-"))
    except InvalidOperation as exc:
        raise ValueError("answer is not a valid decimal") from exc
    if not parsed.is_finite():
        raise ValueError("answer must be finite")
    return parsed


def within_tolerance(
    actual: Decimal,
    expected: Decimal,
    absolute: Decimal,
    relative: Decimal,
) -> bool:
    difference = abs(actual - expected)
    allowed = max(absolute, relative * abs(expected))
    return difference <= allowed

