import random
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
from pydantic import ValidationError

from math_benchmark.enums import (
    CorrectnessScore,
    InstructionFollowingScore,
    ReasoningQualityScore,
    RunStatus,
)
from math_benchmark.schemas import HumanEvaluation, Problem, RunAttempt
from math_benchmark.validators.extraction import extract_final_answer
from math_benchmark.validators.registry import validate_answer


class EvaluationError(ValueError):
    """An annotation file is incomplete, inconsistent, or duplicated."""


ANNOTATION_COLUMNS = [
    "evaluation_id",
    "run_id",
    "problem_id",
    "model_name",
    "category",
    "difficulty",
    "question",
    "reference_solution",
    "raw_response",
    "extracted_answer",
    "extraction_needs_review",
    "automated_result",
    "final_answer_correct",
    "human_override",
    "original_automated_result",
    "override_reason",
    "correctness_score",
    "reasoning_quality_score",
    "instruction_following_score",
    "primary_error",
    "secondary_error",
    "first_error_step",
    "evaluator_confidence",
    "review_notes",
    "reviewer_id",
    "annotation_timestamp",
    "blinded_pass",
]


def derive_fully_correct_solution(evaluation: HumanEvaluation) -> bool:
    return bool(
        evaluation.final_answer_correct is True
        and evaluation.correctness_score == CorrectnessScore.FULLY_CORRECT
        and evaluation.reasoning_quality_score is not None
        and evaluation.reasoning_quality_score >= ReasoningQualityScore.SOUND
        and evaluation.instruction_following_score == InstructionFollowingScore.COMPLETE
    )


def create_annotation_sheet(
    attempts: list[RunAttempt],
    problems: list[Problem],
    output_path: Path,
    technical_failures_path: Path,
) -> pd.DataFrame:
    """Create stable human-review and technical-failure CSV exports."""
    by_id = {problem.problem_id: problem for problem in problems}
    annotation_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    for attempt in attempts:
        if attempt.status != RunStatus.SUCCESS:
            failure_rows.append(attempt.model_dump(mode="json"))
            continue
        try:
            problem = by_id[attempt.problem_id]
        except KeyError as exc:
            raise EvaluationError(f"unknown problem_id {attempt.problem_id}") from exc
        extracted = extract_final_answer(attempt.raw_response or "")
        automated = validate_answer(problem, extracted)
        annotation_rows.append(
            {
                "evaluation_id": f"eval-{attempt.run_id}",
                "run_id": attempt.run_id,
                "problem_id": problem.problem_id,
                "model_name": attempt.model_name,
                "category": problem.category.value,
                "difficulty": problem.difficulty.value,
                "question": problem.question,
                "reference_solution": problem.reference_solution,
                "raw_response": attempt.raw_response,
                "extracted_answer": extracted.text,
                "extraction_needs_review": extracted.needs_review,
                "automated_result": automated.is_correct,
                "final_answer_correct": "",
                "human_override": False,
                "original_automated_result": automated.is_correct,
                "override_reason": "",
                "correctness_score": "",
                "reasoning_quality_score": "",
                "instruction_following_score": "",
                "primary_error": "",
                "secondary_error": "",
                "first_error_step": "",
                "evaluator_confidence": "",
                "review_notes": "",
                "reviewer_id": "",
                "annotation_timestamp": "",
                "blinded_pass": False,
            }
        )
    frame = pd.DataFrame(annotation_rows, columns=ANNOTATION_COLUMNS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, lineterminator="\n")
    technical_failures_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(failure_rows, columns=RunAttempt.model_fields).to_csv(
        technical_failures_path, index=False, lineterminator="\n"
    )
    return frame


def _optional(value: str) -> str | None:
    stripped = value.strip()
    return stripped if stripped else None


def _optional_bool(value: str) -> bool | None:
    normalized = value.strip().lower()
    if not normalized:
        return None
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise EvaluationError(f"invalid boolean value {value!r}")


def _optional_int(value: str) -> int | None:
    stripped = value.strip()
    if not stripped:
        return None
    try:
        return int(stripped)
    except ValueError as exc:
        raise EvaluationError(f"invalid integer value {value!r}") from exc


def _evaluation_from_row(row: dict[str, str]) -> HumanEvaluation:
    bool_fields = {
        "final_answer_correct",
        "automated_result",
        "original_automated_result",
    }
    required_bool_fields = {"human_override", "blinded_pass"}
    int_fields = {
        "correctness_score",
        "reasoning_quality_score",
        "instruction_following_score",
        "evaluator_confidence",
    }
    values: dict[str, Any] = {}
    for field in HumanEvaluation.model_fields:
        raw = row.get(field, "")
        if field in bool_fields:
            values[field] = _optional_bool(raw)
        elif field in required_bool_fields:
            values[field] = _optional_bool(raw) or False
        elif field in int_fields:
            values[field] = _optional_int(raw)
        else:
            values[field] = _optional(raw)
    try:
        return HumanEvaluation.model_validate(values)
    except ValidationError as exc:
        raise EvaluationError(f"invalid evaluation {row.get('evaluation_id', '')}: {exc}") from exc


def load_completed_evaluations(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "evaluation_id" not in raw:
        raise EvaluationError("missing evaluation_id column")
    duplicates = raw["evaluation_id"][raw["evaluation_id"].duplicated()].tolist()
    if duplicates:
        raise EvaluationError(f"duplicate evaluation_id: {duplicates[0]}")
    evaluations = [_evaluation_from_row(row) for row in raw.to_dict(orient="records")]
    records = []
    for evaluation in evaluations:
        record = evaluation.model_dump(mode="json")
        record["fully_correct_solution"] = derive_fully_correct_solution(evaluation)
        records.append(record)
    return pd.DataFrame(records)


def select_blinded_recheck(frame: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    if not 0 < fraction <= 1:
        raise EvaluationError("recheck fraction must be in (0, 1]")
    required = {"run_id", "model_name", "difficulty"}
    if not required.issubset(frame.columns):
        raise EvaluationError(f"recheck frame missing columns: {sorted(required - set(frame.columns))}")
    target = max(1, round(len(frame) * fraction))
    groups = list(frame.groupby(["model_name", "difficulty"], sort=True).groups.values())
    rng = random.Random(seed)
    selected: list[int] = []
    if target >= len(groups):
        for indices in groups:
            selected.append(rng.choice(list(indices)))
    remaining = [index for index in frame.index if index not in selected]
    rng.shuffle(remaining)
    selected.extend(remaining[: max(0, target - len(selected))])
    selected = selected[:target]

    context_columns = [
        column
        for column in (
            "run_id",
            "problem_id",
            "model_name",
            "category",
            "difficulty",
            "question",
            "reference_solution",
            "raw_response",
        )
        if column in frame.columns
    ]
    blinded = frame.loc[selected, context_columns].reset_index(drop=True).copy()
    blinded.insert(0, "blinded_review_id", [f"blind-{number:03d}" for number in range(1, len(blinded) + 1)])
    return blinded
