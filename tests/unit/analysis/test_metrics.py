from pathlib import Path

import pandas as pd
import pytest

from math_benchmark.analysis.metrics import (
    AnalysisError,
    annotation_agreement,
    error_distribution,
    metrics_by_category,
    metrics_by_difficulty,
    overall_metrics,
    wilson_interval,
)

FIXTURE = Path(__file__).parents[2] / "fixtures" / "analysis" / "evaluations.csv"


@pytest.fixture
def evaluation_frame() -> pd.DataFrame:
    return pd.read_csv(FIXTURE)


def test_overall_metrics_are_hand_computed(evaluation_frame: pd.DataFrame) -> None:
    result = overall_metrics(evaluation_frame).set_index("model_name")
    assert result.loc["model-a", "n"] == 2
    assert result.loc["model-a", "final_answer_accuracy"] == pytest.approx(0.5)
    assert result.loc["model-a", "fully_correct_rate"] == pytest.approx(0.5)
    assert result.loc["model-a", "mean_correctness"] == pytest.approx(3.0)


def test_grouped_metrics_carry_counts(evaluation_frame: pd.DataFrame) -> None:
    category = metrics_by_category(evaluation_frame)
    difficulty = metrics_by_difficulty(evaluation_frame)
    assert set(category["n"]) == {1}
    assert all("(n=1)" in label for label in category["category_label"])
    assert set(difficulty["n"]) == {1}
    errors = error_distribution(evaluation_frame)
    assert errors.groupby("model_name")["n"].sum().to_dict() == {"model-a": 2, "model-b": 2}


def test_technical_failures_are_not_in_math_denominator(evaluation_frame: pd.DataFrame) -> None:
    timeout = evaluation_frame.iloc[[0]].copy()
    timeout["run_id"] = "timeout"
    timeout["status"] = "timeout"
    for column in ["final_answer_correct", "fully_correct_solution", "correctness_score"]:
        timeout[column] = pd.NA
    frame = pd.concat([evaluation_frame[evaluation_frame["model_name"] == "model-a"], timeout])
    result = overall_metrics(frame).iloc[0]
    assert result["n_scored"] == 2
    assert result["n_technical_failures"] == 1


@pytest.mark.parametrize(
    ("successes", "total", "expected"),
    [(0, 20, (0.0, 0.1611)), (10, 20, (0.2993, 0.7007)), (20, 20, (0.8389, 1.0))],
)
def test_wilson_interval(successes: int, total: int, expected: tuple[float, float]) -> None:
    actual = wilson_interval(successes, total)
    assert tuple(round(value, 4) for value in actual) == expected


def test_primary_and_stability_cannot_mix(evaluation_frame: pd.DataFrame) -> None:
    stability = evaluation_frame.iloc[[0]].copy()
    stability["experiment_kind"] = "stability"
    with pytest.raises(AnalysisError, match="experiment kinds"):
        overall_metrics(pd.concat([evaluation_frame, stability]))


def test_annotation_agreement_reports_separate_measures() -> None:
    rows = [
        {
            "blinded_review_id": "b1",
            "pass": "original",
            "correctness_score": 4,
            "reasoning_quality_score": 3,
            "instruction_following_score": 2,
            "primary_error": "correct",
            "reviewer_id": "same-reviewer",
        },
        {
            "blinded_review_id": "b1",
            "pass": "recheck",
            "correctness_score": 3,
            "reasoning_quality_score": 3,
            "instruction_following_score": 2,
            "primary_error": "correct",
            "reviewer_id": "same-reviewer",
        },
        {
            "blinded_review_id": "b2",
            "pass": "original",
            "correctness_score": 2,
            "reasoning_quality_score": 2,
            "instruction_following_score": 1,
            "primary_error": "incomplete_reasoning",
            "reviewer_id": "same-reviewer",
        },
        {
            "blinded_review_id": "b2",
            "pass": "recheck",
            "correctness_score": 2,
            "reasoning_quality_score": 1,
            "instruction_following_score": 0,
            "primary_error": "logical_inference_error",
            "reviewer_id": "same-reviewer",
        },
    ]
    result = annotation_agreement(pd.DataFrame(rows)).iloc[0]
    assert result["correctness_exact_agreement"] == pytest.approx(0.5)
    assert result["correctness_within_one_agreement"] == pytest.approx(1.0)
    assert result["reasoning_exact_agreement"] == pytest.approx(0.5)
    assert result["reasoning_within_one_agreement"] == pytest.approx(1.0)
    assert result["instruction_exact_agreement"] == pytest.approx(0.5)
    assert result["primary_error_agreement"] == pytest.approx(0.5)
    assert "cohen_kappa" not in result.index

