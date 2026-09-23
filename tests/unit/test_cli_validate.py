import json
from pathlib import Path

from typer.testing import CliRunner

from math_benchmark.cli import app
from math_benchmark.enums import Category, Difficulty
from tests.unit.test_dataset import make_problem

runner = CliRunner()


def test_validate_command_reports_twenty_valid_problems(tmp_path: Path) -> None:
    layout = [
        (Category.ALGEBRA, [Difficulty.FOUNDATIONAL, Difficulty.INTERMEDIATE, Difficulty.INTERMEDIATE, Difficulty.ADVANCED]),
        (Category.CALCULUS, [Difficulty.FOUNDATIONAL, Difficulty.INTERMEDIATE, Difficulty.INTERMEDIATE, Difficulty.ADVANCED]),
        (Category.PROBABILITY, [Difficulty.FOUNDATIONAL, Difficulty.INTERMEDIATE, Difficulty.ADVANCED]),
        (Category.LOGIC, [Difficulty.FOUNDATIONAL, Difficulty.INTERMEDIATE, Difficulty.ADVANCED]),
        (Category.GRAPH_INTERPRETATION, [Difficulty.FOUNDATIONAL, Difficulty.INTERMEDIATE, Difficulty.ADVANCED]),
        (Category.WORD_PROBLEMS, [Difficulty.INTERMEDIATE, Difficulty.INTERMEDIATE, Difficulty.ADVANCED]),
    ]
    (tmp_path / "figures").mkdir()
    (tmp_path / "sources").mkdir()
    (tmp_path / "figures" / "chart.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (tmp_path / "sources" / "chart.csv").write_text("x,y\n0,0\n", encoding="utf-8")
    records = []
    for category, difficulties in layout:
        for index, difficulty in enumerate(difficulties, 1):
            figure = category == Category.GRAPH_INTERPRETATION
            records.append(
                make_problem(
                    problem_id=f"{category.value[:3].upper()}-{index:03d}",
                    category=category,
                    difficulty=difficulty,
                    figure_path="figures/chart.png" if figure else None,
                    figure_source_path="sources/chart.csv" if figure else None,
                ).model_dump_json()
            )
    dataset = tmp_path / "problems.jsonl"
    dataset.write_text("\n".join(records) + "\n", encoding="utf-8")
    result = runner.invoke(app, ["validate", "--dataset", str(dataset), "--data-root", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "20 problems valid" in result.output
    assert "benchmark version 1.0.0" in result.output


def test_validate_command_reports_malformed_line(tmp_path: Path) -> None:
    dataset = tmp_path / "broken.jsonl"
    dataset.write_text(json.dumps(make_problem().model_dump(mode="json")) + "\n{bad}\n", encoding="utf-8")
    result = runner.invoke(app, ["validate", "--dataset", str(dataset), "--data-root", str(tmp_path)])
    assert result.exit_code != 0
    assert "broken.jsonl:2:" in result.output
