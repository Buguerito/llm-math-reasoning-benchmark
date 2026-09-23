from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from math_benchmark.cli import app
from math_benchmark.enums import (
    CorrectnessScore,
    ErrorCategory,
    InstructionFollowingScore,
    ReasoningQualityScore,
    RunStatus,
)
from math_benchmark.evaluation import (
    EvaluationError,
    create_annotation_sheet,
    derive_fully_correct_solution,
    load_completed_evaluations,
    select_blinded_recheck,
)
from math_benchmark.schemas import HumanEvaluation, RunAttempt
from tests.unit.test_dataset import make_problem


def human_evaluation(**updates) -> HumanEvaluation:
    values = {
        "evaluation_id": "eval-1",
        "run_id": "run-1",
        "final_answer_correct": True,
        "automated_result": True,
        "correctness_score": CorrectnessScore.FULLY_CORRECT,
        "reasoning_quality_score": ReasoningQualityScore.SOUND,
        "instruction_following_score": InstructionFollowingScore.COMPLETE,
        "primary_error": ErrorCategory.CORRECT,
        "evaluator_confidence": 3,
        "review_notes": "Checked against the reference solution.",
        "reviewer_id": "reviewer-1",
        "annotation_timestamp": datetime(2026, 9, 23, tzinfo=UTC),
    }
    values.update(updates)
    return HumanEvaluation(**values)


def test_fully_correct_requires_all_four_conditions() -> None:
    evaluation = human_evaluation()
    assert derive_fully_correct_solution(evaluation) is True
    assert (
        derive_fully_correct_solution(
            evaluation.model_copy(
                update={"instruction_following_score": InstructionFollowingScore.PARTIAL}
            )
        )
        is False
    )


def test_correct_label_conflicts_with_incorrect_answer() -> None:
    with pytest.raises(ValidationError, match="primary_error=correct"):
        human_evaluation(final_answer_correct=False, primary_error="correct")


def test_override_requires_reason_and_original_result() -> None:
    with pytest.raises(ValidationError, match="override_reason"):
        human_evaluation(human_override=True, override_reason=None)


@pytest.mark.parametrize(
    ("field", "value"),
    [("correctness_score", 5), ("reasoning_quality_score", -1), ("instruction_following_score", 3)],
)
def test_rubric_ranges_are_enforced(field: str, value: int) -> None:
    with pytest.raises(ValidationError):
        human_evaluation(**{field: value})


def test_noncorrect_label_requires_first_error_step() -> None:
    with pytest.raises(ValidationError, match="first_error_step"):
        human_evaluation(primary_error="logical_inference_error", first_error_step=None)


def test_correct_label_rejects_secondary_error() -> None:
    with pytest.raises(ValidationError, match="secondary_error"):
        human_evaluation(primary_error="correct", secondary_error="incomplete_reasoning")


def test_completed_evaluation_rejects_blank_reviewer() -> None:
    with pytest.raises(ValidationError, match="reviewer_id"):
        human_evaluation(reviewer_id="   ")


def test_duplicate_evaluation_ids_are_rejected(tmp_path: Path) -> None:
    evaluation = human_evaluation().model_dump(mode="json")
    frame = pd.DataFrame([evaluation, evaluation])
    path = tmp_path / "duplicates.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(EvaluationError, match="duplicate evaluation_id"):
        load_completed_evaluations(path)


def test_completed_fixture_loads_with_derived_field() -> None:
    root = Path(__file__).parents[2]
    frame = load_completed_evaluations(
        root / "tests" / "fixtures" / "evaluations" / "completed.csv"
    )
    assert list(frame["fully_correct_solution"]) == [True, False]


def test_annotation_sheet_separates_technical_failures(tmp_path: Path) -> None:
    problem = make_problem()
    common = {
        "problem_id": problem.problem_id,
        "benchmark_version": "1.0.0",
        "prompt_version": "1.0.0",
        "provider": "mock",
        "model_name": "mock:1",
        "rendered_prompt": "Solve.",
        "started_at": datetime(2026, 1, 1, tzinfo=UTC),
    }
    attempts = [
        RunAttempt(
            run_id="success",
            raw_response="Reasoning.\nFinal answer: 1",
            status=RunStatus.SUCCESS,
            **common,
        ),
        RunAttempt(
            run_id="timeout",
            status=RunStatus.TIMEOUT,
            error_type="ProviderTimeoutError",
            error_message="slow",
            **common,
        ),
    ]
    output = tmp_path / "annotations.csv"
    failures = tmp_path / "technical.csv"
    frame = create_annotation_sheet(attempts, [problem], output, failures)
    assert list(frame["run_id"]) == ["success"]
    assert frame.loc[0, "raw_response"] == "Reasoning.\nFinal answer: 1"
    assert bool(frame.loc[0, "automated_result"]) is True
    technical = pd.read_csv(failures)
    assert list(technical["run_id"]) == ["timeout"]


def test_blinded_recheck_is_stratified_and_hides_annotations() -> None:
    rows = []
    difficulties = ["foundational", "intermediate", "advanced"]
    for index in range(40):
        rows.append(
            {
                "evaluation_id": f"eval-{index}",
                "run_id": f"run-{index}",
                "problem_id": f"P-{index}",
                "model_name": "model-a" if index % 2 == 0 else "model-b",
                "difficulty": difficulties[index % 3],
                "question": f"Question {index}",
                "raw_response": f"Response {index}",
                "correctness_score": index % 5,
                "primary_error": "correct",
                "review_notes": "hidden",
            }
        )
    blinded = select_blinded_recheck(pd.DataFrame(rows), fraction=0.20, seed=20260923)
    assert len(blinded) == 8
    assert blinded["run_id"].is_unique
    assert set(blinded["model_name"]) == {"model-a", "model-b"}
    assert set(blinded["difficulty"]) == set(difficulties)
    assert "blinded_review_id" in blinded
    assert not any(
        token in column
        for column in blinded.columns
        for token in ("score", "error", "note", "automated", "final_answer", "reviewer")
    )


def test_evaluate_cli_writes_annotation_and_failure_files(tmp_path: Path) -> None:
    root = Path(__file__).parents[2]
    output = tmp_path / "annotation_sheet.csv"
    result = CliRunner().invoke(
        app,
        [
            "evaluate",
            "--dataset",
            str(root / "data" / "problems.jsonl"),
            "--attempts",
            str(root / "tests" / "fixtures" / "evaluations" / "attempts.jsonl"),
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    assert len(pd.read_csv(output)) == 2
    assert len(pd.read_csv(tmp_path / "annotation_sheet.technical_failures.csv")) == 1
