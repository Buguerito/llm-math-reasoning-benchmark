from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]


class ReportError(ValueError):
    """A report would misrepresent fixture or incomplete data."""


def _all_fixture(frame: pd.DataFrame) -> bool:
    if "is_fixture" not in frame or frame.empty:
        return False
    return bool(
        frame["is_fixture"]
        .map(lambda value: value is True or str(value).strip().lower() in {"true", "1"})
        .all()
    )


def _csv_block(frame: pd.DataFrame) -> str:
    return "```csv\n" + str(frame.to_csv(index=False, lineterminator="\n")) + "```"


def render_report(
    evaluations: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    plot_paths: list[Path],
    output_path: Path,
    *,
    fixture_report: bool = False,
) -> Path:
    synthetic = _all_fixture(evaluations)
    if synthetic and not fixture_report:
        raise ReportError("synthetic inputs require --fixture-report")
    lines: list[str] = []
    if synthetic:
        lines.extend(["SYNTHETIC TEST DATA — NOT EXPERIMENTAL RESULTS", ""])
    lines.extend(
        [
            "# LLM Mathematical Reasoning Evaluation",
            "",
            "## Methods",
            "",
            "Responses were scored with deterministic final-answer validation and the documented human rubric. Technical failures are reported separately from mathematical denominators.",
            "",
            "## Results",
            "",
            _csv_block(tables["overall"]),
            "",
            "### Category results",
            "",
            _csv_block(tables["category"]),
            "",
            "### Difficulty results",
            "",
            _csv_block(tables["difficulty"]),
            "",
        ]
    )
    for path in plot_paths:
        lines.append(f"![{path.stem}]({path.name})")
    lines.extend(
        [
            "",
            "## Case Studies",
            "",
            "Add representative correct and failure cases only after completing blinded annotation.",
            "",
            "## Stability",
            "",
            "Primary and stability-probe runs are analyzed separately; no continuous trend is inferred from ordinal difficulty labels.",
            "",
            "## Annotation Quality",
            "",
            "Report exact and adjacent-score delayed-recheck agreement. Report Cohen's kappa only for independent reviewers.",
            "",
            "## Limitations",
            "",
            "The benchmark contains 20 authored items, so category estimates are descriptive and have wide uncertainty.",
            "",
            "## Reproduction",
            "",
            "Run `python -m math_benchmark analyze --evaluations <completed.csv> --output <directory>` from the pinned environment.",
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return output_path
