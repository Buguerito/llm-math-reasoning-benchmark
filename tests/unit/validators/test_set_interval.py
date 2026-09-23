import pytest

from math_benchmark.enums import AnswerType, ValidatorKind
from math_benchmark.schemas import GroundTruth, Problem, ValidationSpec
from math_benchmark.validators.models import ExtractedAnswer
from math_benchmark.validators.registry import validate_answer
from math_benchmark.validators.set_interval import set_interval_equivalent
from tests.unit.test_dataset import make_problem


@pytest.fixture
def interval_problem() -> Problem:
    return make_problem().model_copy(
        update={
            "answer_type": AnswerType.SET_INTERVAL,
            "ground_truth": GroundTruth(
                display="(-∞, 2] ∪ [5, ∞)", canonical="(-inf, 2] U [5, inf)"
            ),
            "validation": ValidationSpec(validator=ValidatorKind.SET_INTERVAL),
        }
    )


def test_interval_union_equivalence(interval_problem: Problem) -> None:
    result = validate_answer(
        interval_problem, ExtractedAnswer("[5,infinity) union (-infinity,2]", False, None)
    )
    assert result.is_correct is True


def test_unordered_finite_sets_are_equivalent() -> None:
    assert set_interval_equivalent("{3, 1, 2}", "{1,2,3}").is_correct is True


def test_endpoint_openness_matters(interval_problem: Problem) -> None:
    result = validate_answer(
        interval_problem, ExtractedAnswer("(-inf, 2) U [5, inf)", False, None)
    )
    assert result.is_correct is False


def test_malformed_interval_requires_review(interval_problem: Problem) -> None:
    result = validate_answer(interval_problem, ExtractedAnswer("(1, nope]", False, None))
    assert result.is_correct is None
    assert result.needs_review is True

