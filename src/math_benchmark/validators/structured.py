import json
from typing import Any

from math_benchmark.validators.models import ValidationResult


def _contains_required(actual: Any, expected: Any) -> bool:
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and _contains_required(actual[key], expected_value)
            for key, expected_value in expected.items()
        )
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            _contains_required(actual_value, expected_value)
            for actual_value, expected_value in zip(actual, expected, strict=True)
        )
    return type(actual) is type(expected) and actual == expected


def structured_text_equivalent(actual: str, expected: str) -> ValidationResult:
    try:
        actual_value = json.loads(actual)
        expected_value = json.loads(expected)
    except json.JSONDecodeError:
        return ValidationResult(None, True, actual.strip(), "malformed structured JSON")
    correct = _contains_required(actual_value, expected_value)
    normalized = json.dumps(actual_value, sort_keys=True, separators=(",", ":"))
    return ValidationResult(correct, False, normalized, "required structured fields comparison")

