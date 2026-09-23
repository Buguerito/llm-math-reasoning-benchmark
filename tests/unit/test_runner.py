from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from math_benchmark.enums import ExperimentKind, RunStatus
from math_benchmark.providers.base import (
    ModelConfig,
    ModelIdentity,
    PromptConfig,
    ProviderProtocolError,
    ProviderResponse,
    ProviderTimeoutError,
)
from math_benchmark.runner import (
    RunCompatibilityError,
    RunManifest,
    RunSettings,
    build_run_plan,
    run_benchmark,
)
from math_benchmark.schemas import RunAttempt
from math_benchmark.storage import JsonlAttemptStore
from tests.unit.test_dataset import make_problem


def settings() -> RunSettings:
    return RunSettings(
        prompt=PromptConfig(version="1.0.0", base_instruction="Solve.\nFinal answer: <answer>"),
        models=(ModelConfig(tag="mock:1", provider="mock"),),
        profile="primary",
        experiment_kind=ExperimentKind.PRIMARY,
    )


def previous(status: RunStatus, attempt_number: int = 1) -> RunAttempt:
    return RunAttempt(
        run_id=f"old-{status}-{attempt_number}",
        problem_id="ALG-I-002",
        benchmark_version="1.0.0",
        prompt_version="1.0.0",
        provider="mock",
        model_name="mock:1",
        model_digest="sha256:mock",
        generation_parameters={"seed": 42},
        rendered_prompt="Solve.",
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        status=status,
        attempt_number=attempt_number,
    )


def manifest(attempts: list[RunAttempt], benchmark_version: str = "1.0.0") -> RunManifest:
    return RunManifest(
        benchmark_version=benchmark_version,
        prompt_version="1.0.0",
        profile="primary",
        experiment_kind=ExperimentKind.PRIMARY,
        attempts=attempts,
    )


def test_resume_rejects_version_mismatch() -> None:
    problem_v2 = make_problem().model_copy(update={"benchmark_version": "2.0.0"})
    with pytest.raises(RunCompatibilityError, match="benchmark version"):
        build_run_plan([problem_v2], settings(), existing_manifest=manifest([]))


def test_resume_rejects_prompt_version_mismatch() -> None:
    existing = manifest([]).model_copy(update={"prompt_version": "0.9.0"})
    with pytest.raises(RunCompatibilityError, match="prompt version"):
        build_run_plan([make_problem()], settings(), existing_manifest=existing)


def test_resume_skips_success_and_retries_timeout() -> None:
    problems = [make_problem("ALG-F-001"), make_problem("ALG-I-002")]
    attempts = [
        previous(RunStatus.SUCCESS).model_copy(update={"problem_id": "ALG-F-001"}),
        previous(RunStatus.TIMEOUT),
    ]
    plan = build_run_plan(problems, settings(), existing_manifest=manifest(attempts))
    assert [(item.problem.problem_id, item.attempt_number) for item in plan.items] == [
        ("ALG-I-002", 2)
    ]
    assert plan.items[0].parent_run_id == attempts[1].run_id


class SequenceProvider:
    def __init__(self, outcomes: Iterator[str | Exception]) -> None:
        self.outcomes = outcomes
        self.calls = 0

    def resolve_model(self, model: ModelConfig) -> ModelIdentity:
        return ModelIdentity(model.tag, "sha256:mock")

    def generate(self, problem, prompt, model) -> ProviderResponse:
        self.calls += 1
        outcome = next(self.outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        now = datetime(2026, 1, 1, tzinfo=UTC)
        return ProviderResponse(
            raw_text=outcome,
            provider=model.provider,
            model_tag=model.tag,
            started_at=now,
            completed_at=now,
            rendered_prompt=f"{prompt.base_instruction}\n\n{problem.question}",
            image_paths=(),
            generation_parameters=model.generation_options(),
        )


def test_timeout_gets_one_linked_retry(tmp_path: Path) -> None:
    provider = SequenceProvider(iter([ProviderTimeoutError("slow"), "Final answer: 1"]))
    store = JsonlAttemptStore(tmp_path / "attempts.jsonl")
    plan = build_run_plan([make_problem()], settings())
    attempts = run_benchmark(plan, provider, {"mock:1": ModelIdentity("mock:1", "sha256:mock")}, store)
    assert [attempt.status for attempt in attempts] == [RunStatus.TIMEOUT, RunStatus.SUCCESS]
    assert attempts[1].parent_run_id == attempts[0].run_id
    assert attempts[1].attempt_number == 2


def test_invalid_provider_content_is_not_retried(tmp_path: Path) -> None:
    provider = SequenceProvider(iter([ProviderProtocolError("bad content")]))
    attempts = run_benchmark(
        build_run_plan([make_problem()], settings()),
        provider,
        {"mock:1": ModelIdentity("mock:1", "sha256:mock")},
        JsonlAttemptStore(tmp_path / "attempts.jsonl"),
    )
    assert provider.calls == 1
    assert attempts[0].status == RunStatus.INVALID_RESPONSE


def test_missing_final_answer_marker_is_still_success(tmp_path: Path) -> None:
    provider = SequenceProvider(iter(["I cannot determine it."]))
    attempts = run_benchmark(
        build_run_plan([make_problem()], settings()),
        provider,
        {"mock:1": ModelIdentity("mock:1", "sha256:mock")},
        JsonlAttemptStore(tmp_path / "attempts.jsonl"),
    )
    assert attempts[0].status == RunStatus.SUCCESS
    assert attempts[0].raw_response == "I cannot determine it."

