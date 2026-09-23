from collections.abc import Callable
from decimal import Decimal
from fractions import Fraction

from math_benchmark.enums import AnswerType
from math_benchmark.schemas import Problem
from math_benchmark.validators.exact import normalize_exact, parse_fraction
from math_benchmark.validators.models import ExtractedAnswer, ValidationResult
from math_benchmark.validators.numeric import parse_decimal, within_tolerance

Validator = Callable[[Problem, str], tuple[bool, str, str]]


def _expected_values(problem: Problem) -> list[str]:
    return [problem.ground_truth.canonical, *problem.ground_truth.accepted_equivalents]


def _validate_exact(problem: Problem, actual: str) -> tuple[bool, str, str]:
    normalized = normalize_exact(actual)
    expected = {normalize_exact(value) for value in _expected_values(problem)}
    return normalized in expected, normalized, "exact normalized comparison"


def _validate_rational(problem: Problem, actual: str) -> tuple[bool, str, str]:
    parsed = parse_fraction(actual)
    expected: set[Fraction] = {parse_fraction(value) for value in _expected_values(problem)}
    return parsed in expected, str(parsed), "exact rational comparison"


def _validate_numeric(problem: Problem, actual: str) -> tuple[bool, str, str]:
    parsed = parse_decimal(actual)
    expected = parse_decimal(problem.ground_truth.canonical)
    absolute = Decimal(str(problem.validation.absolute_tolerance))
    relative = Decimal(str(problem.validation.relative_tolerance))
    correct = within_tolerance(parsed, expected, absolute, relative)
    return correct, str(parsed), "numeric tolerance comparison"


VALIDATORS: dict[AnswerType, Validator] = {
    AnswerType.EXACT: _validate_exact,
    AnswerType.RATIONAL: _validate_rational,
    AnswerType.NUMERIC: _validate_numeric,
}


def validate_answer(problem: Problem, extracted: ExtractedAnswer) -> ValidationResult:
    """Validate an extracted answer without allowing user data to raise."""
    if extracted.text is None:
        return ValidationResult(None, True, None, extracted.reason or "no extracted answer")

    validator = VALIDATORS.get(problem.answer_type)
    if validator is None:
        return ValidationResult(None, True, normalize_exact(extracted.text), "unsupported answer type")

    try:
        correct, normalized, reason = validator(problem, extracted.text)
    except (ValueError, ZeroDivisionError):
        return ValidationResult(None, True, normalize_exact(extracted.text), "unparseable answer")

    if extracted.needs_review:
        reason = f"{reason}; {extracted.reason or 'extraction requires review'}"
    return ValidationResult(correct, extracted.needs_review, normalized, reason)
