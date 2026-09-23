import hashlib
from collections import Counter
from pathlib import Path

import pytest

from math_benchmark.dataset import (
    EXPECTED_CATEGORY_COUNTS,
    EXPECTED_DIFFICULTY_COUNTS,
    DatasetValidationError,
    load_problems,
    validate_benchmark,
)
from math_benchmark.schemas import Problem
from math_benchmark.validators.models import ExtractedAnswer
from math_benchmark.validators.registry import validate_answer
from scripts.build_dataset import build_dataset

ROOT = Path(__file__).parents[2]
DATA_ROOT = ROOT / "data"
PROBLEM_SOURCE_DIR = DATA_ROOT / "problem_sources"
CANONICAL_PATH = DATA_ROOT / "problems.jsonl"
EXPECTED_FIGURE_HASHES = {
    "GRF-A-003.png": "90269cf03d0e824d9ea0a8377b973c7f232b55c9b6a6aa6a1a13cb8f51f938ff",
    "GRF-F-001.png": "370e951490fedf8f4cab2196a56baf422c367ae3d3c7c7173c3ea4a6447b7114",
    "GRF-I-002.png": "f6c50b3ac2fc436f7892850903a8634f81b4bb067fd1f82af8ce51068ccea695",
}


@pytest.fixture(scope="module")
def canonical_problems() -> list[Problem]:
    return load_problems(CANONICAL_PATH)


def test_canonical_distribution_and_copy(canonical_problems: list[Problem]) -> None:
    validate_benchmark(canonical_problems, DATA_ROOT)
    assert len(canonical_problems) == 20
    assert len({problem.problem_id for problem in canonical_problems}) == 20
    assert Counter(problem.category for problem in canonical_problems) == EXPECTED_CATEGORY_COUNTS
    assert Counter(problem.difficulty for problem in canonical_problems) == EXPECTED_DIFFICULTY_COUNTS
    assert all(problem.question.strip() and problem.reference_solution.strip() for problem in canonical_problems)
    assert all(problem.source.source_type.value in {"authored", "adapted", "public"} for problem in canonical_problems)


def test_every_ground_truth_is_accepted(canonical_problems: list[Problem]) -> None:
    for problem in canonical_problems:
        extracted = ExtractedAnswer(problem.ground_truth.canonical, False, None)
        result = validate_answer(problem, extracted)
        assert result.is_correct is True, f"{problem.problem_id}: {result.reason}"


def test_figure_hashes_are_frozen() -> None:
    observed = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((DATA_ROOT / "figures").glob("*.png"))
    }
    assert observed == EXPECTED_FIGURE_HASHES


def test_build_is_byte_for_byte_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    build_dataset(PROBLEM_SOURCE_DIR, first)
    build_dataset(PROBLEM_SOURCE_DIR, second)
    assert first.read_bytes() == second.read_bytes()
    assert first.read_bytes() == CANONICAL_PATH.read_bytes()


def test_build_refuses_cross_version_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "existing.jsonl"
    changed = canonical_problems_for_version("2.0.0")
    output.write_text("\n".join(problem.model_dump_json() for problem in changed) + "\n", encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="--new-version"):
        build_dataset(PROBLEM_SOURCE_DIR, output)


def canonical_problems_for_version(version: str) -> list[Problem]:
    return [problem.model_copy(update={"benchmark_version": version}) for problem in load_problems(CANONICAL_PATH)]
