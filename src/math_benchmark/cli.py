from collections import Counter
from pathlib import Path
from typing import Annotated

import typer

from math_benchmark.dataset import DatasetValidationError, load_problems, validate_benchmark

app = typer.Typer(help="Evaluate mathematical reasoning with local language models.")


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
