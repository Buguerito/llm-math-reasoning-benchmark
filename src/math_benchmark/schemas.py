from datetime import UTC, datetime
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from math_benchmark.enums import (
    AnswerType,
    Category,
    CorrectnessScore,
    Difficulty,
    ErrorCategory,
    ExperimentKind,
    InstructionFollowingScore,
    ReasoningQualityScore,
    RunStatus,
    SourceType,
    ValidatorKind,
)

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class GroundTruth(StrictModel):
    display: NonEmptyStr
    canonical: NonEmptyStr
    accepted_equivalents: list[str] = Field(default_factory=list)


class ValidationSpec(StrictModel):
    validator: ValidatorKind
    absolute_tolerance: float | None = Field(default=None, ge=0)
    relative_tolerance: float | None = Field(default=None, ge=0)
    options: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_numeric_tolerances(self) -> "ValidationSpec":
        if self.validator == ValidatorKind.NUMERIC and (
            self.absolute_tolerance is None or self.relative_tolerance is None
        ):
            raise ValueError("numeric validator requires absolute and relative tolerance")
        return self


class SourceInfo(StrictModel):
    source_type: SourceType
    reference: str | None = None
    transformation: str | None = None


class Problem(StrictModel):
    problem_id: NonEmptyStr
    category: Category
    subdomain: NonEmptyStr
    difficulty: Difficulty
    question: NonEmptyStr
    answer_type: AnswerType
    ground_truth: GroundTruth
    reference_solution: NonEmptyStr
    validation: ValidationSpec
    skills_tested: list[NonEmptyStr] = Field(min_length=1)
    figure_path: str | None = None
    figure_source_path: str | None = None
    source: SourceInfo
    verification_notes: NonEmptyStr
    benchmark_version: NonEmptyStr

    @model_validator(mode="after")
    def require_figure_pair(self) -> "Problem":
        if bool(self.figure_path) != bool(self.figure_source_path):
            raise ValueError("figure_path and figure_source_path must appear together")
        return self


class RunAttempt(StrictModel):
    run_id: NonEmptyStr
    parent_run_id: str | None = None
    problem_id: NonEmptyStr
    benchmark_version: NonEmptyStr
    prompt_version: NonEmptyStr
    provider: NonEmptyStr
    model_name: NonEmptyStr
    model_digest: str | None = None
    generation_parameters: dict[str, Any] = Field(default_factory=dict)
    rendered_prompt: NonEmptyStr
    image_paths: list[str] = Field(default_factory=list)
    input_hashes: dict[str, str] = Field(default_factory=dict)
    raw_response: str | None = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    latency_seconds: float | None = Field(default=None, ge=0)
    attempt_number: int = Field(default=1, ge=1)
    status: RunStatus
    error_type: str | None = None
    error_message: str | None = None
    experiment_kind: ExperimentKind = ExperimentKind.PRIMARY


class HumanEvaluation(StrictModel):
    evaluation_id: NonEmptyStr
    run_id: NonEmptyStr
    final_answer_correct: bool | None = None
    automated_result: bool | None = None
    human_override: bool = False
    original_automated_result: bool | None = None
    override_reason: str | None = None
    correctness_score: CorrectnessScore | None = None
    reasoning_quality_score: ReasoningQualityScore | None = None
    instruction_following_score: InstructionFollowingScore | None = None
    primary_error: ErrorCategory | None = None
    secondary_error: ErrorCategory | None = None
    first_error_step: str | None = None
    evaluator_confidence: int | None = Field(default=None, ge=1, le=3)
    review_notes: str | None = None
    reviewer_id: str | None = None
    annotation_timestamp: datetime | None = None
    blinded_pass: bool = False
