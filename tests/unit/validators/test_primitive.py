from decimal import Decimal

import pytest

from math_benchmark.enums import AnswerType, ValidatorKind
from math_benchmark.schemas import GroundTruth, Problem, ValidationSpec
from math_benchmark.validators.models import ExtractedAnswer
from math_benchmark.validators.numeric import within_tolerance
from math_benchmark.validators.registry import validate_answer
from tests.unit.test_dataset import make_problem


@pytest.fixture
def rational_problem() -> Problem:
    return make_problem().model_copy(
        update={
            "answer_type": AnswerType.RATIONAL,
            "ground_truth": GroundTruth(display="6", canonical="6"),
            "validation": ValidationSpec(validator=ValidatorKind.EXACT),
        }
    )


@pytest.fixture
def decimal_problem() -> Problem:
    return make_problem().model_copy(
        update={
            "answer_type": AnswerType.NUMERIC,
            "ground_truth": GroundTruth(display="1/3", canonical="0.3333333333"),
            "validation": ValidationSpec(
                validator=ValidatorKind.NUMERIC,
                absolute_tolerance=0.001,
                relative_tolerance=0.0,
            ),
        }
    )


def test_exact_normalization() -> None:
    problem = make_problem().model_copy(
        update={
            "answer_type": AnswerType.EXACT,
            "ground_truth": GroundTruth(display="x = -6", canonical="x = -6"),
        }
    )
    result = validate_answer(problem, ExtractedAnswer(" $x   = −6$. ", False, None))
    assert result.is_correct is True
    assert result.normalized_actual == "x = -6"


@pytest.mark.parametrize("actual", ["6", "6.0", "12/2", "+6"])
def test_rational_equivalence(actual: str, rational_problem: Problem) -> None:
    result = validate_answer(rational_problem, ExtractedAnswer(actual, False, None))
    assert result.is_correct is True


def test_numeric_tolerance_is_explicit(decimal_problem: Problem) -> None:
    assert validate_answer(decimal_problem, ExtractedAnswer("0.3334", False, None)).is_correct
    assert not validate_answer(decimal_problem, ExtractedAnswer("0.34", False, None)).is_correct


@pytest.mark.parametrize("actual", ["nan", "inf", "-inf", "six"])
def test_invalid_numeric_answers_require_review(actual: str, decimal_problem: Problem) -> None:
    result = validate_answer(decimal_problem, ExtractedAnswer(actual, False, None))
    assert result.is_correct is None
    assert result.needs_review is True


def test_missing_extracted_answer_requires_review(decimal_problem: Problem) -> None:
    result = validate_answer(decimal_problem, ExtractedAnswer(None, True, "empty response"))
    assert result.is_correct is None
    assert result.needs_review is True


def test_decimal_tolerance_uses_larger_allowance() -> None:
    assert within_tolerance(Decimal(105), Decimal(100), Decimal(1), Decimal("0.05"))


def test_unsupported_answer_type_requires_review() -> None:
    problem = make_problem().model_copy(update={"answer_type": AnswerType.SYMBOLIC})
    result = validate_answer(problem, ExtractedAnswer("x", False, None))
    assert result.is_correct is None
    assert result.needs_review is True
