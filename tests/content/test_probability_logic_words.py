from pathlib import Path

import pytest

from math_benchmark.dataset import load_problems
from math_benchmark.schemas import Problem
from math_benchmark.validators.models import ExtractedAnswer
from math_benchmark.validators.registry import validate_answer

DATA_ROOT = Path(__file__).parents[2] / "data" / "problem_sources"

MANIFEST = {
    "PROB-F-001": ("foundational", "rational", "1/3"),
    "PROB-I-002": ("intermediate", "rational", "2/3"),
    "PROB-A-003": ("advanced", "exact", "8"),
    "LOG-F-001": (
        "foundational",
        "structured_text",
        '{"valid":true,"rule":"modus tollens"}',
    ),
    "LOG-I-002": (
        "intermediate",
        "structured_text",
        '{"exists_epsilon_gt_0":true,"for_every_N":true,"exists_n_gte_N":true,"condition":"|a_n-L|>=epsilon"}',
    ),
    "LOG-A-003": (
        "advanced",
        "structured_text",
        '{"A":"knave","B":"knight","C":"knave"}',
    ),
    "WORD-I-001": ("intermediate", "numeric", "30"),
    "WORD-I-002": ("intermediate", "rational", "15/4"),
    "WORD-A-003": ("advanced", "symbolic", "20*log(5)"),
}


@pytest.fixture(scope="module")
def authored_problems() -> list[Problem]:
    filenames = ["probability.jsonl", "logic.jsonl", "word_problems.jsonl"]
    return [problem for filename in filenames for problem in load_problems(DATA_ROOT / filename)]


def test_probability_logic_word_manifest(authored_problems: list[Problem]) -> None:
    observed = {
        problem.problem_id: (
            problem.difficulty.value,
            problem.answer_type.value,
            problem.ground_truth.canonical,
        )
        for problem in authored_problems
    }
    assert observed == MANIFEST


def test_each_category_has_three_reviewed_self_validating_records(
    authored_problems: list[Problem],
) -> None:
    assert {category: sum(problem.category.value == category for problem in authored_problems) for category in ("probability", "logic", "word_problems")} == {
        "probability": 3,
        "logic": 3,
        "word_problems": 3,
    }
    for problem in authored_problems:
        assert len(problem.reference_solution) >= 100
        assert "verified" in problem.verification_notes.lower()
        result = validate_answer(
            problem, ExtractedAnswer(problem.ground_truth.canonical, False, None)
        )
        assert result.is_correct is True, (problem.problem_id, result)


def test_hht_uses_explicit_states(authored_problems: list[Problem]) -> None:
    problem = next(problem for problem in authored_problems if problem.problem_id == "PROB-A-003")
    assert all(state in problem.reference_solution for state in ('state ""', 'state "H"', 'state "HH"'))


def test_knights_check_covers_all_assignments(authored_problems: list[Problem]) -> None:
    problem = next(problem for problem in authored_problems if problem.problem_id == "LOG-A-003")
    assert "all eight" in problem.verification_notes.lower()
