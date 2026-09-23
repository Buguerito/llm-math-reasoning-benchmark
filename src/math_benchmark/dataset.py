from collections import Counter
from collections.abc import Sequence
from pathlib import Path, PureWindowsPath

from pydantic import ValidationError

from math_benchmark.enums import Category, Difficulty
from math_benchmark.schemas import Problem

EXPECTED_CATEGORY_COUNTS = {
    Category.ALGEBRA: 4,
    Category.CALCULUS: 4,
    Category.PROBABILITY: 3,
    Category.LOGIC: 3,
    Category.GRAPH_INTERPRETATION: 3,
    Category.WORD_PROBLEMS: 3,
}
EXPECTED_DIFFICULTY_COUNTS = {
    Difficulty.FOUNDATIONAL: 5,
    Difficulty.INTERMEDIATE: 9,
    Difficulty.ADVANCED: 6,
}


class DatasetValidationError(ValueError):
    """Raised when benchmark data is malformed or internally inconsistent."""


def load_problems(path: Path) -> list[Problem]:
    problems: list[Problem] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise DatasetValidationError(f"{path}: {exc}") from exc
    for line_number, raw_line in enumerate(lines, 1):
        if not raw_line.strip():
            continue
        try:
            problems.append(Problem.model_validate_json(raw_line))
        except (ValueError, ValidationError) as exc:
            raise DatasetValidationError(f"{path}:{line_number}: {exc}") from exc
    if not problems:
        raise DatasetValidationError(f"{path}: dataset is empty")
    return problems


def _validated_asset_path(data_root: Path, relative_path: str, problem_id: str) -> Path:
    candidate = Path(relative_path)
    root = data_root.resolve()
    if candidate.is_absolute() or PureWindowsPath(relative_path).is_absolute():
        raise DatasetValidationError(
            f"{problem_id}: asset path {relative_path!r} must remain inside data root"
        )
    resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(root):
        raise DatasetValidationError(
            f"{problem_id}: asset path {relative_path!r} must remain inside data root"
        )
    if not resolved.is_file():
        raise DatasetValidationError(f"{problem_id}: asset {relative_path!r} does not exist")
    return resolved


def validate_benchmark(
    problems: Sequence[Problem],
    data_root: Path,
    enforce_distribution: bool = True,
) -> None:
    seen: set[str] = set()
    for problem in problems:
        if problem.problem_id in seen:
            raise DatasetValidationError(f"duplicate problem_id {problem.problem_id}")
        seen.add(problem.problem_id)
        if problem.figure_path is not None and problem.figure_source_path is not None:
            _validated_asset_path(data_root, problem.figure_path, problem.problem_id)
            _validated_asset_path(data_root, problem.figure_source_path, problem.problem_id)

    versions = {problem.benchmark_version for problem in problems}
    if len(versions) != 1:
        raise DatasetValidationError(f"benchmark versions must match; found {sorted(versions)}")

    if not enforce_distribution:
        return
    category_counts = Counter(problem.category for problem in problems)
    difficulty_counts = Counter(problem.difficulty for problem in problems)
    if dict(category_counts) != EXPECTED_CATEGORY_COUNTS:
        raise DatasetValidationError(
            f"category distribution mismatch: expected {EXPECTED_CATEGORY_COUNTS}, got {dict(category_counts)}"
        )
    if dict(difficulty_counts) != EXPECTED_DIFFICULTY_COUNTS:
        raise DatasetValidationError(
            "difficulty distribution mismatch: "
            f"expected {EXPECTED_DIFFICULTY_COUNTS}, got {dict(difficulty_counts)}"
        )
