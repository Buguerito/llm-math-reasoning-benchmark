import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from math_benchmark.enums import ExperimentKind, RunStatus
from math_benchmark.environment import EnvironmentSnapshot
from math_benchmark.providers.base import (
    ModelConfig,
    ModelIdentity,
    PromptConfig,
    Provider,
    ProviderError,
    ProviderProtocolError,
    ProviderTimeoutError,
    render_prompt,
)
from math_benchmark.schemas import Problem, RunAttempt
from math_benchmark.storage import JsonlAttemptStore


class RunCompatibilityError(ValueError):
    """A resumed run does not match the current immutable inputs."""


class RunManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    manifest_version: str = "1.0.0"
    benchmark_version: str
    prompt_version: str
    profile: str
    experiment_kind: ExperimentKind
    model_identities: list[dict[str, str]] = Field(default_factory=list)
    environment: EnvironmentSnapshot | None = None
    attempts: list[RunAttempt] = Field(default_factory=list)


@dataclass(frozen=True)
class RunSettings:
    prompt: PromptConfig
    models: tuple[ModelConfig, ...]
    profile: str
    experiment_kind: ExperimentKind


@dataclass(frozen=True)
class RunPlanItem:
    problem: Problem
    model: ModelConfig
    attempt_number: int = 1
    parent_run_id: str | None = None


@dataclass(frozen=True)
class RunPlan:
    items: tuple[RunPlanItem, ...]
    settings: RunSettings
    benchmark_version: str


def _attempt_key(problem_id: str, model_name: str, seed: int) -> tuple[str, str, int]:
    return problem_id, model_name, seed


def build_run_plan(
    problems: list[Problem],
    settings: RunSettings,
    *,
    existing_manifest: RunManifest | None = None,
) -> RunPlan:
    if not problems:
        raise RunCompatibilityError("run requires at least one problem")
    versions = {problem.benchmark_version for problem in problems}
    if len(versions) != 1:
        raise RunCompatibilityError("benchmark versions do not match")
    benchmark_version = versions.pop()
    if existing_manifest is not None:
        if existing_manifest.benchmark_version != benchmark_version:
            raise RunCompatibilityError("benchmark version does not match existing manifest")
        if existing_manifest.prompt_version != settings.prompt.version:
            raise RunCompatibilityError("prompt version does not match existing manifest")

    existing_by_key: dict[tuple[str, str, int], list[RunAttempt]] = {}
    for attempt in existing_manifest.attempts if existing_manifest else []:
        seed = int(attempt.generation_parameters.get("seed", 42))
        existing_by_key.setdefault(_attempt_key(attempt.problem_id, attempt.model_name, seed), []).append(
            attempt
        )

    items: list[RunPlanItem] = []
    retryable = {RunStatus.TIMEOUT, RunStatus.PROVIDER_ERROR}
    for model in settings.models:
        for problem in problems:
            previous = sorted(
                existing_by_key.get(_attempt_key(problem.problem_id, model.tag, model.seed), []),
                key=lambda item: item.attempt_number,
            )
            if not previous:
                items.append(RunPlanItem(problem, model))
                continue
            latest = previous[-1]
            if latest.status == RunStatus.SUCCESS:
                continue
            if latest.status in retryable and latest.attempt_number < 2:
                items.append(
                    RunPlanItem(problem, model, latest.attempt_number + 1, latest.run_id)
                )
    return RunPlan(tuple(items), settings, benchmark_version)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _input_hashes(problem: Problem, rendered_prompt: str, data_root: Path) -> dict[str, str]:
    hashes = {"prompt_sha256": _sha256(rendered_prompt.encode("utf-8"))}
    if problem.figure_path:
        path = Path(problem.figure_path)
        path = path if path.is_absolute() else data_root / path
        hashes["image_sha256"] = _sha256(path.read_bytes())
    return hashes


def _failure_status(exc: ProviderError) -> RunStatus:
    if isinstance(exc, ProviderTimeoutError):
        return RunStatus.TIMEOUT
    if isinstance(exc, ProviderProtocolError):
        return RunStatus.INVALID_RESPONSE
    return RunStatus.PROVIDER_ERROR


def run_benchmark(
    plan: RunPlan,
    provider: Provider,
    identities: dict[str, ModelIdentity],
    store: JsonlAttemptStore,
    *,
    data_root: Path = Path("data"),
) -> list[RunAttempt]:
    """Execute a plan, appending immutable attempts and one bounded retry."""
    recorded: list[RunAttempt] = []
    for item in plan.items:
        attempt_number = item.attempt_number
        parent_run_id = item.parent_run_id
        while True:
            run_id = str(uuid4())
            rendered = render_prompt(item.problem, plan.settings.prompt)
            try:
                response = provider.generate(item.problem, plan.settings.prompt, item.model)
                attempt = RunAttempt(
                    run_id=run_id,
                    parent_run_id=parent_run_id,
                    problem_id=item.problem.problem_id,
                    benchmark_version=item.problem.benchmark_version,
                    prompt_version=plan.settings.prompt.version,
                    provider=item.model.provider,
                    model_name=item.model.tag,
                    model_digest=identities[item.model.tag].digest,
                    generation_parameters=item.model.generation_options(),
                    rendered_prompt=response.rendered_prompt,
                    image_paths=list(response.image_paths),
                    input_hashes=_input_hashes(item.problem, response.rendered_prompt, data_root),
                    raw_response=response.raw_text,
                    started_at=response.started_at,
                    completed_at=response.completed_at,
                    latency_seconds=response.latency_seconds,
                    attempt_number=attempt_number,
                    status=RunStatus.SUCCESS,
                    experiment_kind=plan.settings.experiment_kind,
                )
            except ProviderError as exc:
                status = _failure_status(exc)
                now = datetime.now(UTC)
                attempt = RunAttempt(
                    run_id=run_id,
                    parent_run_id=parent_run_id,
                    problem_id=item.problem.problem_id,
                    benchmark_version=item.problem.benchmark_version,
                    prompt_version=plan.settings.prompt.version,
                    provider=item.model.provider,
                    model_name=item.model.tag,
                    model_digest=identities[item.model.tag].digest,
                    generation_parameters=item.model.generation_options(),
                    rendered_prompt=rendered,
                    image_paths=list(filter(None, [item.problem.figure_path])),
                    input_hashes=_input_hashes(item.problem, rendered, data_root),
                    started_at=now,
                    completed_at=now,
                    latency_seconds=0.0,
                    attempt_number=attempt_number,
                    status=status,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    experiment_kind=plan.settings.experiment_kind,
                )
            store.append(attempt)
            recorded.append(attempt)
            if attempt.status in {RunStatus.TIMEOUT, RunStatus.PROVIDER_ERROR} and attempt_number < 2:
                parent_run_id = attempt.run_id
                attempt_number += 1
                continue
            break
    return recorded


def manifest_for_run(
    plan: RunPlan,
    identities: dict[str, ModelIdentity],
    attempts: list[RunAttempt],
    environment: EnvironmentSnapshot | None,
) -> RunManifest:
    identity_values = sorted(
        ({"tag": identity.tag, "digest": identity.digest} for identity in identities.values()),
        key=lambda value: value["tag"],
    )
    return RunManifest(
        benchmark_version=plan.benchmark_version,
        prompt_version=plan.settings.prompt.version,
        profile=plan.settings.profile,
        experiment_kind=plan.settings.experiment_kind,
        model_identities=identity_values,
        environment=environment,
        attempts=attempts,
    )
