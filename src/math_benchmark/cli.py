from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import httpx
import typer
import yaml  # type: ignore[import-untyped]

from math_benchmark.dataset import DatasetValidationError, load_problems, validate_benchmark
from math_benchmark.enums import ExperimentKind
from math_benchmark.environment import collect_environment, system_command_runner
from math_benchmark.providers.base import ModelConfig, ModelIdentity, PromptConfig, ProviderError
from math_benchmark.providers.ollama import OllamaProvider
from math_benchmark.runner import (
    RunCompatibilityError,
    RunManifest,
    RunSettings,
    build_run_plan,
    manifest_for_run,
    run_benchmark,
)
from math_benchmark.storage import JsonlAttemptStore, write_manifest_atomic

app = typer.Typer(help="Evaluate mathematical reasoning with local language models.")
ROOT = Path(__file__).parents[2]
STABILITY_IDS = ("ALG-I-003", "CAL-A-004", "PROB-A-003", "LOG-I-002", "GRF-A-003")


@app.callback()
def main() -> None:
    """Run benchmark validation, inference, evaluation, and analysis commands."""


@app.command("validate")
def validate_command(
    dataset: Annotated[Path, typer.Option(exists=False)] = Path("data/problems.jsonl"),
    data_root: Annotated[Path, typer.Option(exists=False)] = Path("data"),
) -> None:
    """Validate the benchmark schema, assets, versions, and distribution."""
    try:
        problems = load_problems(dataset)
        validate_benchmark(problems, data_root)
    except DatasetValidationError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    category_counts = Counter(problem.category.value for problem in problems)
    difficulty_counts = Counter(problem.difficulty.value for problem in problems)
    typer.echo(f"{len(problems)} problems valid; benchmark version {problems[0].benchmark_version}")
    typer.echo(f"categories: {dict(sorted(category_counts.items()))}")
    typer.echo(f"difficulties: {dict(sorted(difficulty_counts.items()))}")


def _yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DatasetValidationError(f"{path}: expected a YAML object")
    return value


def _settings(model_tags: list[str], profile: str) -> RunSettings:
    model_document = _yaml(ROOT / "configs" / "models.yaml")
    prompt_document = _yaml(ROOT / "configs" / "prompts.yaml")
    available = {entry["tag"]: entry for entry in model_document["models"]}
    unknown = sorted(set(model_tags) - set(available))
    if unknown:
        raise DatasetValidationError(f"unknown configured model tags: {unknown}")
    try:
        profile_data = model_document["profiles"][profile]
    except KeyError as exc:
        raise DatasetValidationError(f"unknown profile {profile!r}") from exc
    seeds = profile_data["seeds"]
    models = tuple(
        ModelConfig(
            tag=tag,
            provider=available[tag]["provider"],
            temperature=float(profile_data["temperature"]),
            seed=int(seed),
            num_ctx=int(profile_data["num_ctx"]),
            num_predict=int(profile_data["num_predict"]),
            timeout_seconds=float(profile_data["timeout_seconds"]),
        )
        for tag in model_tags
        for seed in seeds
    )
    prompt = PromptConfig(
        version=str(prompt_document["version"]),
        base_instruction=str(prompt_document["base_instruction"]),
    )
    kind = ExperimentKind.STABILITY if profile == "stability" else ExperimentKind.PRIMARY
    return RunSettings(prompt=prompt, models=models, profile=profile, experiment_kind=kind)


@app.command("run")
def run_command(
    models: Annotated[list[str] | None, typer.Option("--models")] = None,
    profile: Annotated[str, typer.Option()] = "primary",
    dataset: Annotated[Path, typer.Option(exists=False)] = Path("data/problems.jsonl"),
    output: Annotated[Path | None, typer.Option()] = None,
    manifest: Annotated[Path | None, typer.Option()] = None,
    problem_id: Annotated[list[str] | None, typer.Option("--problem-id")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    base_url: Annotated[str, typer.Option()] = "http://localhost:11434",
) -> None:
    """Plan or execute an immutable local-model benchmark run."""
    selected_models = models or ["qwen2.5vl:3b", "gemma3:4b"]
    try:
        problems = load_problems(dataset)
        validate_benchmark(problems, dataset.parent)
        settings = _settings(selected_models, profile)
        selected_ids = set(problem_id or (STABILITY_IDS if profile == "stability" else ()))
        if selected_ids:
            known_ids = {problem.problem_id for problem in problems}
            missing = sorted(selected_ids - known_ids)
            if missing:
                raise DatasetValidationError(f"unknown problem IDs: {missing}")
            problems = [problem for problem in problems if problem.problem_id in selected_ids]
        plan = build_run_plan(problems, settings)
    except (DatasetValidationError, RunCompatibilityError, KeyError, TypeError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(
        f"{len(plan.items)} planned attempts; experiment_kind={settings.experiment_kind.value}"
    )
    if dry_run:
        return

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_path = output or Path(f"results/raw/{profile}-{timestamp}.jsonl")
    manifest_path = manifest or output_path.with_suffix(".manifest.json")
    store = JsonlAttemptStore(output_path)
    existing_attempts = store.load_all()
    if manifest_path.exists():
        existing = RunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        existing = existing.model_copy(update={"attempts": existing_attempts})
        plan = build_run_plan(problems, settings, existing_manifest=existing)

    try:
        with httpx.Client() as client:
            provider = OllamaProvider(base_url, client, data_root=dataset.parent)
            identities: dict[str, ModelIdentity] = {}
            for model in settings.models:
                if model.tag not in identities:
                    identities[model.tag] = provider.resolve_model(model)
            environment = collect_environment(system_command_runner)
            initial_manifest = manifest_for_run(
                plan, identities, existing_attempts, environment
            )
            write_manifest_atomic(manifest_path, initial_manifest)
            attempts = run_benchmark(plan, provider, identities, store, data_root=dataset.parent)
    except (ProviderError, OSError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    all_attempts = existing_attempts + attempts
    run_manifest = manifest_for_run(plan, identities, all_attempts, environment)
    write_manifest_atomic(manifest_path, run_manifest)
    typer.echo(f"wrote {len(attempts)} attempts to {output_path}")
