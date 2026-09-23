import csv
import hashlib
from pathlib import Path

import matplotlib.image as mpimg

from math_benchmark.dataset import load_problems
from math_benchmark.validators.models import ExtractedAnswer
from math_benchmark.validators.registry import validate_answer
from scripts.generate_figures import generate_all

ROOT = Path(__file__).parents[2]
SOURCE_DIR = ROOT / "data" / "figure_source_data"

EXPECTED = {
    "GRF-F-001": {
        "rows": [("Jan", 12), ("Feb", 15), ("Mar", 15), ("Apr", 21), ("May", 18)],
        "answer": {"largest_increase": 6, "from": "Mar", "to": "Apr"},
    },
    "GRF-I-002": {
        "rows": [(0, 0), (2, 6), (5, 12), (6, 12)],
        "answer": {"fastest_interval": [0, 2], "speed": 3, "overall_average_speed": 2},
    },
    "GRF-A-003": {
        "rows": [(0, -2), (2, 0), (4, 2), (6, 0)],
        "answer": {"minimum_x": 2, "net_change_0_to_6": 2},
    },
}


def _read_rows(problem_id: str) -> list[tuple[str | int, int]]:
    with (SOURCE_DIR / f"{problem_id}.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))[1:]
    first_is_text = problem_id == "GRF-F-001"
    return [(row[0] if first_is_text else int(row[0]), int(row[1])) for row in rows]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_source_data_is_locked() -> None:
    for problem_id, expected in EXPECTED.items():
        assert _read_rows(problem_id) == expected["rows"]


def test_generation_is_deterministic_and_large_enough(tmp_path: Path) -> None:
    first_dir, second_dir = tmp_path / "first", tmp_path / "second"
    first = generate_all(SOURCE_DIR, first_dir)
    second = generate_all(SOURCE_DIR, second_dir)
    assert [path.name for path in first] == [f"{problem_id}.png" for problem_id in EXPECTED]
    assert [path.name for path in second] == [path.name for path in first]
    for first_path, second_path in zip(first, second, strict=True):
        assert mpimg.imread(first_path).shape[:2] == (600, 960)
        assert mpimg.imread(second_path).shape[:2] == (600, 960)
        assert _sha256(first_path) == _sha256(second_path)


def test_graph_records_match_assets_and_answers() -> None:
    problems = load_problems(ROOT / "data" / "problem_sources" / "graph_interpretation.jsonl")
    assert [problem.problem_id for problem in problems] == list(EXPECTED)
    for problem in problems:
        assert problem.figure_path == f"figures/{problem.problem_id}.png"
        assert problem.figure_source_path == f"figure_source_data/{problem.problem_id}.csv"
        assert (ROOT / "data" / problem.figure_path).is_file()
        assert (ROOT / "data" / problem.figure_source_path).is_file()
        result = validate_answer(
            problem, ExtractedAnswer(problem.ground_truth.canonical, False, None)
        )
        assert result.is_correct is True
