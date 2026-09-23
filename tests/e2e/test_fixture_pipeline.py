from pathlib import Path

from math_benchmark.analysis.metrics import (
    error_distribution,
    metrics_by_category,
    metrics_by_difficulty,
    overall_metrics,
)
from math_benchmark.analysis.report import render_report
from math_benchmark.dataset import load_problems
from math_benchmark.enums import ExperimentKind
from math_benchmark.evaluation import create_annotation_sheet, load_completed_evaluations
from math_benchmark.providers.base import ModelConfig, ModelIdentity, PromptConfig
from math_benchmark.providers.mock import MockProvider
from math_benchmark.runner import RunSettings, build_run_plan, run_benchmark
from math_benchmark.storage import JsonlAttemptStore

ROOT = Path(__file__).parents[2]


def test_fixture_pipeline_runs_without_network(tmp_path: Path) -> None:
    all_problems = load_problems(ROOT / "data" / "problems.jsonl")
    selected_ids = {"ALG-F-001", "CAL-I-003"}
    problems = [problem for problem in all_problems if problem.problem_id in selected_ids]
    provider = MockProvider(
        {
            "ALG-F-001": "Clear denominators and solve.\nFinal answer: x = 6",
            "CAL-I-003": "Use the second-order expansion.\nFinal answer: 2",
        }
    )
    model = ModelConfig(tag="mock:1", provider="mock")
    settings = RunSettings(
        prompt=PromptConfig(
            version="1.0.0",
            base_instruction="Solve concisely.\nEnd with Final answer: <answer>",
        ),
        models=(model,),
        profile="fixture",
        experiment_kind=ExperimentKind.PRIMARY,
    )
    attempts = run_benchmark(
        build_run_plan(problems, settings),
        provider,
        {"mock:1": ModelIdentity("mock:1", "sha256:mock")},
        JsonlAttemptStore(tmp_path / "attempts.jsonl"),
        data_root=ROOT / "data",
    )
    annotation_path = tmp_path / "annotations.csv"
    create_annotation_sheet(
        attempts,
        problems,
        annotation_path,
        tmp_path / "technical_failures.csv",
    )

    completed = load_completed_evaluations(
        ROOT / "tests" / "fixtures" / "evaluations" / "completed.csv"
    )
    completed["model_name"] = "mock:1"
    completed["category"] = ["algebra", "calculus"]
    completed["difficulty"] = ["foundational", "intermediate"]
    completed["status"] = "success"
    completed["experiment_kind"] = "primary"
    completed["is_fixture"] = True
    tables = {
        "overall": overall_metrics(completed),
        "category": metrics_by_category(completed),
        "difficulty": metrics_by_difficulty(completed),
        "errors": error_distribution(completed),
    }
    report = render_report(
        completed,
        tables,
        [],
        tmp_path / "report.md",
        fixture_report=True,
    )
    assert len(attempts) == 2
    assert annotation_path.exists()
    assert report.exists()

