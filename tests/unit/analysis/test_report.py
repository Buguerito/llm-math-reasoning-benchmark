from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from math_benchmark.analysis.metrics import (
    error_distribution,
    metrics_by_category,
    metrics_by_difficulty,
    overall_metrics,
)
from math_benchmark.analysis.report import ReportError, render_report
from math_benchmark.cli import app

FIXTURE = Path(__file__).parents[2] / "fixtures" / "analysis" / "evaluations.csv"


def tables(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "overall": overall_metrics(frame),
        "category": metrics_by_category(frame),
        "difficulty": metrics_by_difficulty(frame),
        "errors": error_distribution(frame),
    }


def test_fixture_report_has_warning_and_required_headings(tmp_path: Path) -> None:
    frame = pd.read_csv(FIXTURE)
    output = tmp_path / "report.md"
    render_report(frame, tables(frame), [], output, fixture_report=True)
    text = output.read_text(encoding="utf-8")
    assert text.startswith("SYNTHETIC TEST DATA — NOT EXPERIMENTAL RESULTS")
    for heading in [
        "Methods",
        "Results",
        "Case Studies",
        "Stability",
        "Annotation Quality",
        "Limitations",
        "Reproduction",
    ]:
        assert f"## {heading}" in text


def test_synthetic_data_requires_fixture_mode(tmp_path: Path) -> None:
    frame = pd.read_csv(FIXTURE)
    with pytest.raises(ReportError, match="fixture-report"):
        render_report(frame, tables(frame), [], tmp_path / "report.md")


def test_analyze_cli_creates_tables_plots_and_report(tmp_path: Path) -> None:
    output = tmp_path / "report"
    result = CliRunner().invoke(
        app,
        [
            "analyze",
            "--evaluations",
            str(FIXTURE),
            "--output",
            str(output),
            "--fixture-report",
        ],
    )
    assert result.exit_code == 0, result.output
    assert len(list(output.glob("*.png"))) == 5
    assert len(list(output.glob("*_metrics.csv"))) == 4
    assert (output / "report.md").is_file()
