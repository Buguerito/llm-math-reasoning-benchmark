from typer.testing import CliRunner

from math_benchmark.cli import app

runner = CliRunner()


def test_primary_dry_run_plans_40_without_network() -> None:
    result = runner.invoke(
        app,
        ["run", "--dry-run", "--models", "qwen2.5vl:3b", "--models", "gemma3:4b"],
    )
    assert result.exit_code == 0, result.output
    assert "40 planned attempts" in result.output


def test_stability_dry_run_plans_30() -> None:
    args = [
        "run",
        "--dry-run",
        "--profile",
        "stability",
        "--models",
        "qwen2.5vl:3b",
        "--models",
        "gemma3:4b",
    ]
    for problem_id in ["ALG-I-003", "CAL-A-004", "PROB-A-003", "LOG-I-002", "GRF-A-003"]:
        args.extend(["--problem-id", problem_id])
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "30 planned attempts" in result.output
    assert "experiment_kind=stability" in result.output
