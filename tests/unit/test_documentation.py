import json
import re
from pathlib import Path

from typer.testing import CliRunner

from math_benchmark.cli import app

ROOT = Path(__file__).parents[2]


def test_required_portfolio_files_exist() -> None:
    for relative in [
        "README.md",
        "docs/methodology.md",
        "docs/target-pc-runbook.md",
        "docs/annotation_guide.md",
        "notebooks/benchmark_analysis.ipynb",
        ".github/workflows/ci.yml",
        "LICENSE",
    ]:
        assert (ROOT / relative).is_file(), relative


def test_readme_has_required_positioning_and_valid_cli_commands() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    lowered = text.lower()
    for phrase in ["20 problems", "human evaluation", "ollama", "limitations", "not experimental results"]:
        assert phrase in lowered
    help_text = CliRunner().invoke(app, ["--help"]).output
    commands = set(re.findall(r"python -m math_benchmark (\w+)", text))
    assert commands == {"validate", "run", "evaluate", "analyze"}
    assert all(command in help_text for command in commands)


def test_readme_relative_links_exist() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    links = re.findall(r"\[[^]]+\]\((?!https?://|#)([^)]+)\)", text)
    assert links
    for link in links:
        path = link.split("#", 1)[0]
        assert (ROOT / path).exists(), link


def test_notebook_is_thin_cleared_and_labels_fixture_data() -> None:
    notebook = json.loads(
        (ROOT / "notebooks" / "benchmark_analysis.ipynb").read_text(encoding="utf-8")
    )
    assert "synthetic" in notebook["cells"][0]["source"][0].lower()
    assert "milestone b" in "".join(notebook["cells"][0]["source"]).lower()
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells
    assert all(cell["execution_count"] is None and cell["outputs"] == [] for cell in code_cells)
    assert all("math_benchmark" in "".join(cell["source"]) or "EVALUATIONS" in "".join(cell["source"]) for cell in code_cells)


def test_ci_is_network_free_during_tests() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert 'pytest -m "not integration"' in workflow
    assert "ollama pull" not in workflow.lower()
    assert "python scripts/build_dataset.py" in workflow
    assert "python scripts/generate_figures.py" not in workflow
    assert "git diff --exit-code data/problems.jsonl" in workflow
