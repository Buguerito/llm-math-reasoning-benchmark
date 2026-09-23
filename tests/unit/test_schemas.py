import pytest
from pydantic import ValidationError

from math_benchmark.enums import AnswerType, Category, Difficulty
from math_benchmark.schemas import GroundTruth, Problem, SourceInfo, ValidationSpec


def valid_problem() -> Problem:
    return Problem(
        problem_id="ALG-F-001",
        category=Category.ALGEBRA,
        subdomain="linear equations",
        difficulty=Difficulty.FOUNDATIONAL,
        question="Solve (3/4)(x - 2) + (1/3)(x + 6) = 7.",
        answer_type=AnswerType.RATIONAL,
        ground_truth=GroundTruth(display="x = 6", canonical="6"),
        reference_solution="Multiply by 12, simplify to 13x = 78, and obtain x = 6.",
        validation=ValidationSpec(
            validator="numeric",
            absolute_tolerance=0.0,
            relative_tolerance=0.0,
        ),
        skills_tested=["equation solving", "fraction arithmetic"],
        source=SourceInfo(source_type="authored", reference=None),
        verification_notes="Independently substituted x = 6 into the original equation.",
        benchmark_version="1.0.0",
    )


def test_valid_problem_round_trips() -> None:
    item = valid_problem()
    assert Problem.model_validate_json(item.model_dump_json()) == item


def test_unknown_category_is_rejected() -> None:
    payload = valid_problem().model_dump(mode="json")
    payload["category"] = "geometry"
    with pytest.raises(ValidationError):
        Problem.model_validate(payload)


def test_numeric_validator_requires_tolerances() -> None:
    payload = valid_problem().model_dump(mode="json")
    payload["validation"] = {"validator": "numeric"}
    with pytest.raises(ValidationError, match="tolerance"):
        Problem.model_validate(payload)


def test_figure_and_figure_source_must_appear_together() -> None:
    payload = valid_problem().model_dump(mode="json")
    payload["figure_path"] = "figures/chart.png"
    with pytest.raises(ValidationError, match="figure_source_path"):
        Problem.model_validate(payload)
