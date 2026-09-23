from pathlib import Path

import pytest

from math_benchmark.dataset import load_problems
from math_benchmark.validators.models import ExtractedAnswer
from math_benchmark.validators.registry import validate_answer

DATA_ROOT = Path(__file__).parents[2] / "data" / "problem_sources"

MANIFEST = {
    "ALG-F-001": ("foundational", "exact", "x = 6"),
    "ALG-I-002": (
        "intermediate",
        "structured_text",
        '{"points":[{"x":"3-sqrt(5)","y":"5-2*sqrt(5)"},{"x":"3+sqrt(5)","y":"5+2*sqrt(5)"}]}',
    ),
    "ALG-I-003": ("intermediate", "set_interval", "{-2, 4}"),
    "ALG-A-004": ("advanced", "symbolic", "x**3 + 2"),
    "CAL-F-001": (
        "foundational",
        "structured_text",
        '{"increasing":"(-inf,-1) U (3,inf)","decreasing":"(-1,3)","local_max":{"x":-1,"y":10},"local_min":{"x":3,"y":-22}}',
    ),
    "CAL-I-002": ("intermediate", "symbolic", "(E - 1)/2"),
    "CAL-I-003": ("intermediate", "exact", "2"),
    "CAL-A-004": (
        "advanced",
        "structured_text",
        '{"half_width":2,"height":8,"maximum_area":32}',
    ),
}


@pytest.fixture(scope="module")
def authored_problems():
    return load_problems(DATA_ROOT / "algebra.jsonl") + load_problems(
        DATA_ROOT / "calculus.jsonl"
    )


def test_algebra_calculus_manifest(authored_problems) -> None:
    observed = {
        problem.problem_id: (
            problem.difficulty.value,
            problem.answer_type.value,
            problem.ground_truth.canonical,
        )
        for problem in authored_problems
    }
    assert observed == MANIFEST


def test_algebra_records_are_complete_and_self_validating(authored_problems) -> None:
    algebra = [problem for problem in authored_problems if problem.problem_id.startswith("ALG-")]
    assert len(algebra) == 4
    for problem in algebra:
        assert len(problem.reference_solution) >= 100
        assert "verified" in problem.verification_notes.lower()
        result = validate_answer(
            problem, ExtractedAnswer(problem.ground_truth.canonical, False, None)
        )
        assert result.is_correct is True, (problem.problem_id, result)


def test_calculus_records_are_complete_and_self_validating(authored_problems) -> None:
    calculus = [problem for problem in authored_problems if problem.problem_id.startswith("CAL-")]
    assert len(calculus) == 4
    for problem in calculus:
        assert len(problem.reference_solution) >= 100
        assert "verified" in problem.verification_notes.lower()
        result = validate_answer(
            problem, ExtractedAnswer(problem.ground_truth.canonical, False, None)
        )
        assert result.is_correct is True, (problem.problem_id, result)
