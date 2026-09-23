import pytest

from math_benchmark.enums import AnswerType, ValidatorKind
from math_benchmark.schemas import GroundTruth, Problem, ValidationSpec
from math_benchmark.validators.bounded import BoundedExecutionResult
from math_benchmark.validators.models import ExtractedAnswer
from math_benchmark.validators.registry import validate_answer
from math_benchmark.validators.symbolic import symbolic_equivalent
from tests.unit.test_dataset import make_problem


@pytest.fixture
def symbolic_problem() -> Problem:
    return make_problem().model_copy(
        update={
            "answer_type": AnswerType.SYMBOLIC,
            "ground_truth": GroundTruth(display="(x+1)^2", canonical="(x + 1)**2"),
            "validation": ValidationSpec(
                validator=ValidatorKind.SYMBOLIC,
                options={"symbols": ["x"], "timeout_seconds": 2.0},
            ),
        }
    )


@pytest.mark.parametrize(
    "unsafe",
    ["__import__('os').system('whoami')", "open('secret.txt').read()", "x.__class__"],
)
def test_symbolic_payloads_outside_allowlist_require_review(
    unsafe: str, symbolic_problem: Problem
) -> None:
    result = validate_answer(symbolic_problem, ExtractedAnswer(unsafe, False, None))
    assert result.is_correct is None
    assert result.needs_review is True
    assert "unsupported symbolic syntax" in result.reason


def test_symbolic_equivalence() -> None:
    result = symbolic_equivalent("(x + 1)**2", "x**2 + 2*x + 1", {"symbols": ["x"]}, 2.0)
    assert result.is_correct is True


def test_symbolic_input_length_is_bounded(symbolic_problem: Problem) -> None:
    result = validate_answer(symbolic_problem, ExtractedAnswer("x+" * 300, False, None))
    assert result.needs_review is True
    assert result.reason == "symbolic input exceeds 512 characters"


def test_worker_timeout_returns_review(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "math_benchmark.validators.symbolic.run_bounded",
        lambda *args, **kwargs: BoundedExecutionResult.timed_out(),
    )
    result = symbolic_equivalent("x", "x", {"symbols": ["x"]}, 0.01)
    assert result.needs_review is True
    assert result.reason == "symbolic validation timed out"

