"""Build the canonical benchmark JSONL from reviewed category sources."""

import argparse
import json
from pathlib import Path

from math_benchmark.dataset import DatasetValidationError, load_problems, validate_benchmark
from math_benchmark.schemas import Problem

ROOT = Path(__file__).parents[1]
SOURCE_FILENAMES = (
    "algebra.jsonl",
    "calculus.jsonl",
    "probability.jsonl",
    "logic.jsonl",
    "graph_interpretation.jsonl",
    "word_problems.jsonl",
)


def _version(problems: list[Problem]) -> str:
    versions = {problem.benchmark_version for problem in problems}
    if len(versions) != 1:
        raise DatasetValidationError(f"benchmark versions must match; found {sorted(versions)}")
    return versions.pop()


def build_dataset(
    source_dir: Path,
    output_path: Path,
    *,
    new_version: bool = False,
) -> list[Problem]:
    """Validate, deterministically merge, and write category source records."""
    data_root = source_dir.parent
    merged: list[Problem] = []
    for filename in SOURCE_FILENAMES:
        problems = sorted(load_problems(source_dir / filename), key=lambda item: item.problem_id)
        validate_benchmark(problems, data_root, enforce_distribution=False)
        merged.extend(problems)

    validate_benchmark(merged, data_root)
    incoming_version = _version(merged)
    if output_path.exists():
        existing_version = _version(load_problems(output_path))
        if existing_version != incoming_version and not new_version:
            raise DatasetValidationError(
                f"refusing to replace benchmark {existing_version} with {incoming_version}; "
                "pass --new-version to confirm the version change"
            )

    lines = [
        json.dumps(
            problem.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        for problem in merged
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-version", action="store_true")
    args = parser.parse_args()
    problems = build_dataset(
        ROOT / "data" / "problem_sources",
        ROOT / "data" / "problems.jsonl",
        new_version=args.new_version,
    )
    print(f"{len(problems)} valid problems at benchmark version {_version(problems)}")


if __name__ == "__main__":
    main()
