import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from math_benchmark.dataset import DatasetValidationError, load_problems, validate_benchmark
from math_benchmark.enums import AnswerType, Category, Difficulty
from math_benchmark.schemas import GroundTruth, Problem, SourceInfo, ValidationSpec


def make_problem(
    problem_id: str = "ALG-F-001",
    category: Category = Category.ALGEBRA,
    difficulty: Difficulty = Difficulty.FOUNDATIONAL,
    figure_path: str | None = None,
    figure_source_path: str | None = None,
) -> Problem:
    return Problem(
        problem_id=problem_id,
        category=category,
        subdomain="test fixture",
        difficulty=difficulty,
        question="TEST FIXTURE — NOT BENCHMARK DATA. Solve the stated problem.",
        answer_type=AnswerType.EXACT,
        ground_truth=GroundTruth(display="1", canonical="1"),
        reference_solution="TEST FIXTURE — The answer follows from the supplied premise.",
        validation=ValidationSpec(validator="exact"),
        skills_tested=["fixture validation"],
        figure_path=figure_path,
        figure_source_path=figure_source_path,
        source=SourceInfo(source_type="authored"),
        verification_notes="TEST FIXTURE — manually checked.",
        benchmark_version="1.0.0",
    )


@pytest.fixture
def valid_twenty(tmp_path: Path) -> Iterator[tuple[list[Problem], Path]]:
    (tmp_path / "figures").mkdir()
    (tmp_path / "figure_source_data").mkdir()
    (tmp_path / "figures" / "chart.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (tmp_path / "figure_source_data" / "chart.csv").write_text("x,y\n0,0\n", encoding="utf-8")
    layout = [
        (Category.ALGEBRA, [Difficulty.FOUNDATIONAL, Difficulty.INTERMEDIATE, Difficulty.INTERMEDIATE, Difficulty.ADVANCED]),
        (Category.CALCULUS, [Difficulty.FOUNDATIONAL, Difficulty.INTERMEDIATE, Difficulty.INTERMEDIATE, Difficulty.ADVANCED]),
        (Category.PROBABILITY, [Difficulty.FOUNDATIONAL, Difficulty.INTERMEDIATE, Difficulty.ADVANCED]),
        (Category.LOGIC, [Difficulty.FOUNDATIONAL, Difficulty.INTERMEDIATE, Difficulty.ADVANCED]),
        (Category.GRAPH_INTERPRETATION, [Difficulty.FOUNDATIONAL, Difficulty.INTERMEDIATE, Difficulty.ADVANCED]),
        (Category.WORD_PROBLEMS, [Difficulty.INTERMEDIATE, Difficulty.INTERMEDIATE, Difficulty.ADVANCED]),
    ]
    problems: list[Problem] = []
    for category, difficulties in layout:
        for index, difficulty in enumerate(difficulties, 1):
            figure = category == Category.GRAPH_INTERPRETATION
            problems.append(
                make_problem(
                    problem_id=f"{category.value[:3].upper()}-{index:03d}",
                    category=category,
                    difficulty=difficulty,
                    figure_path="figures/chart.png" if figure else None,
                    figure_source_path="figure_source_data/chart.csv" if figure else None,
                )
            )
    yield problems, tmp_path


def test_malformed_json_reports_line_number(tmp_path: Path) -> None:
    path = tmp_path / "problems.jsonl"
    path.write_text("\n{not-json}\n", encoding="utf-8")
    with pytest.raises(DatasetValidationError, match=r"problems\.jsonl:2:"):
        load_problems(path)


def test_duplicate_problem_ids_are_rejected(tmp_path: Path) -> None:
    item = make_problem().model_dump(mode="json")
    path = tmp_path / "problems.jsonl"
    path.write_text("\n".join([json.dumps(item), json.dumps(item)]), encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="duplicate problem_id ALG-F-001"):
        validate_benchmark(load_problems(path), tmp_path, enforce_distribution=False)


@pytest.mark.parametrize("bad_path", ["../secret.png", "C:/secret.png"])
def test_figure_paths_cannot_escape_data_root(tmp_path: Path, bad_path: str) -> None:
    problem = make_problem(figure_path=bad_path, figure_source_path="chart.csv")
    with pytest.raises(DatasetValidationError, match="must remain inside data root"):
        validate_benchmark([problem], tmp_path, enforce_distribution=False)


def test_missing_figure_is_rejected(tmp_path: Path) -> None:
    problem = make_problem(figure_path="figures/missing.png", figure_source_path="source.csv")
    with pytest.raises(DatasetValidationError, match="does not exist"):
        validate_benchmark([problem], tmp_path, enforce_distribution=False)


def test_exact_distribution_is_required(valid_twenty: tuple[list[Problem], Path]) -> None:
    problems, root = valid_twenty
    validate_benchmark(problems, root)
    with pytest.raises(DatasetValidationError, match="category distribution"):
        validate_benchmark(problems[:-1], root)

