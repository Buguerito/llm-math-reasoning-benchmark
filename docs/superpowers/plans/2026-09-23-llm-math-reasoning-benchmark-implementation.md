# LLM Mathematical Reasoning Evaluation Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a portfolio-quality, local-first Python benchmark containing 20 verified mathematics problems, an Ollama inference pipeline, deterministic and human evaluation, and reproducible Pandas reports.

**Architecture:** Keep dataset validation, model providers, raw-run storage, answer validation, human evaluation, and analysis as independent typed modules. The current PC must support every development and test command without Ollama; real inference remains a later target-PC milestone.

**Tech Stack:** Python 3.11+, Pydantic 2, Typer, HTTPX, PyYAML, SymPy, Pandas, Matplotlib, Seaborn, pytest, Ruff, mypy, nbformat, GitHub Actions, Ollama HTTP API.

**Spec:** `docs/superpowers/specs/2026-09-23-llm-math-reasoning-benchmark-design.md`

## Global Constraints

- All user-facing repository content, problems, rubrics, documentation, and report copy must be in English.
- Python 3.11 is the minimum supported version.
- The frozen benchmark contains exactly 20 items: Algebra 4, Calculus 4, Probability 3, Logic 3, Graph Interpretation 3, and Word Problems 3.
- Difficulty totals are exactly 5 foundational, 9 intermediate, and 6 advanced.
- The primary models are exactly `qwen2.5vl:3b` and `gemma3:4b`; store exact Ollama digests during inference.
- Do not install Ollama or model weights on the current development PC.
- Do not require paid APIs, API keys, a GPU, or network access for unit tests, dataset validation, annotation, or analysis.
- Keep raw model attempts append-only. Retries create new linked attempts and never overwrite prior output.
- Use deterministic validation plus human evaluation; an LLM must never be the final judge.
- Never present fixtures, mocks, pilots, or synthetic outputs as experimental findings.
- Primary-run results and stability-probe results must remain separate.
- No single composite model score is allowed.
- The portfolio is empirically complete only after real inference and human annotation on the target PC.
- Publish to GitHub only after the non-Ollama test suite passes and the user approves the public repository settings.

## Review Focus

- A figure path that escapes the repository or points to a missing file must fail dataset validation; Task 2 adds both tests.
- Conflicting or repeated `Final answer:` markers must trigger human review instead of silently selecting a favorable answer; Task 3 adds this test.
- Untrusted symbolic text that contains names outside the allowlist or exceeds resource limits must return `needs_review`, never execute; Task 4 adds injection, length, and timeout tests.
- Resuming a partially failed run must skip successful attempts but must reject a benchmark or prompt version mismatch; Task 10 adds both tests.
- An incomplete or contradictory human annotation must be rejected before metrics are calculated; Task 11 adds rubric-range, error-label, and override-consistency tests.

---

## File Map

| Path | Responsibility |
|---|---|
| `pyproject.toml` | Package metadata, dependencies, entry point, and tool settings |
| `.gitignore` | Exclude environments, caches, secrets, and uncommitted run artifacts |
| `configs/models.yaml` | Reproducible model and generation settings |
| `configs/prompts.yaml` | Versioned shared prompt text |
| `data/problem_sources/*.jsonl` | Reviewed category-level source records |
| `data/problems.jsonl` | Deterministically built canonical benchmark |
| `data/figures/*.png` | Three generated graph inputs |
| `data/figure_source_data/*.csv` | Exact data used to generate each graph |
| `src/math_benchmark/enums.py` | Shared closed vocabularies |
| `src/math_benchmark/schemas.py` | Pydantic records for problems, runs, and evaluations |
| `src/math_benchmark/dataset.py` | Dataset loading and integrity validation |
| `src/math_benchmark/validators/` | Extraction and deterministic answer comparison |
| `src/math_benchmark/providers/` | Provider protocol, mock provider, and Ollama adapter |
| `src/math_benchmark/storage.py` | Append-only JSONL storage and resumability |
| `src/math_benchmark/runner.py` | Run planning, retries, and provider orchestration |
| `src/math_benchmark/evaluation.py` | Automated validation and human annotation import/export |
| `src/math_benchmark/analysis/` | Metrics, confidence intervals, charts, and report rendering |
| `src/math_benchmark/cli.py` | Typer commands: validate, run, evaluate, analyze |
| `scripts/build_dataset.py` | Deterministically merge reviewed problem sources |
| `scripts/generate_figures.py` | Recreate graph images from committed CSV files |
| `notebooks/benchmark_analysis.ipynb` | Thin presentation notebook calling package functions |
| `docs/methodology.md` | Experimental and statistical method |
| `docs/annotation_guide.md` | Rubric decision rules and examples |
| `docs/target-pc-runbook.md` | Ollama installation, smoke test, full run, and report steps |
| `.github/workflows/ci.yml` | Network-free lint, type, unit, and dataset checks |
| `tests/` | Unit, integration, fixtures, and end-to-end smoke tests |

---

### Task 1: Package Foundation and Typed Schemas

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `src/math_benchmark/__init__.py`
- Create: `src/math_benchmark/__main__.py`
- Create: `src/math_benchmark/enums.py`
- Create: `src/math_benchmark/schemas.py`
- Create: `tests/unit/test_schemas.py`

**Interfaces:**
- Consumes: none.
- Produces: `Problem`, `GroundTruth`, `ValidationSpec`, `SourceInfo`, `RunAttempt`, `HumanEvaluation`, and all enums imported by later tasks.

- [ ] **Step 1: Add the failing schema tests**

Create `tests/unit/test_schemas.py` with explicit factories and these behaviors:

```python
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from math_benchmark.enums import AnswerType, Category, Difficulty
from math_benchmark.schemas import GroundTruth, Problem, SourceInfo, ValidationSpec


def valid_problem() -> Problem:
    return Problem(
        problem_id="ALG-F-001",
        category=Category.ALGEBRA,
        subdomain="linear equations",
        difficulty=Difficulty.FOUNDATIONAL,
        question="Solve (3/4)(x - 2) + (1/3)(x + 6) = 7.",
        answer_type=AnswerType.RATIONAL,
        ground_truth=GroundTruth(display="x = 6", canonical="6"),
        reference_solution="Multiply by 12, simplify to 13x = 78, and obtain x = 6.",
        validation=ValidationSpec(validator="numeric", absolute_tolerance=0.0, relative_tolerance=0.0),
        skills_tested=["equation solving", "fraction arithmetic"],
        source=SourceInfo(source_type="authored", reference=None),
        verification_notes="Independently substituted x = 6 into the original equation.",
        benchmark_version="1.0.0",
    )


def test_valid_problem_round_trips() -> None:
    item = valid_problem()
    assert Problem.model_validate_json(item.model_dump_json()) == item


def test_unknown_category_is_rejected() -> None:
    payload = valid_problem().model_dump(mode="json")
    payload["category"] = "geometry"
    with pytest.raises(ValidationError):
        Problem.model_validate(payload)


def test_numeric_validator_requires_tolerances() -> None:
    payload = valid_problem().model_dump(mode="json")
    payload["validation"] = {"validator": "numeric"}
    with pytest.raises(ValidationError, match="tolerance"):
        Problem.model_validate(payload)


def test_figure_and_figure_source_must_appear_together() -> None:
    payload = valid_problem().model_dump(mode="json")
    payload["figure_path"] = "figures/chart.png"
    with pytest.raises(ValidationError, match="figure_source_path"):
        Problem.model_validate(payload)
```

- [ ] **Step 2: Run the schema test and verify the import failure**

Run: `python -m pytest tests/unit/test_schemas.py -v`

Expected: FAIL during collection because `math_benchmark` does not exist.

- [ ] **Step 3: Create the package configuration**

Create `pyproject.toml` with a `src` layout, a `math-benchmark = "math_benchmark.cli:app"` script, and these bounded dependencies:

```toml
[build-system]
requires = ["hatchling>=1.25"]
build-backend = "hatchling.build"

[project]
name = "llm-math-reasoning-benchmark"
version = "0.1.0"
description = "A local-first benchmark for evaluating LLM mathematical reasoning."
requires-python = ">=3.11"
dependencies = [
  "httpx>=0.27,<1",
  "matplotlib>=3.9,<4",
  "pandas>=2.2,<4",
  "pydantic>=2.8,<3",
  "PyYAML>=6.0,<7",
  "seaborn>=0.13,<1",
  "sympy>=1.13,<2",
  "typer>=0.12,<1",
]

[project.optional-dependencies]
dev = [
  "mypy>=1.11,<2",
  "nbformat>=5.10,<6",
  "pytest>=8.3,<10",
  "pytest-cov>=5,<8",
  "ruff>=0.6,<1",
]

[project.scripts]
math-benchmark = "math_benchmark.cli:app"

[tool.hatch.build.targets.wheel]
packages = ["src/math_benchmark"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["integration: requires a running Ollama service"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.mypy]
python_version = "3.11"
strict = true
packages = ["math_benchmark"]
```

Add `.gitignore` entries for `.venv/`, `__pycache__/`, `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`, `.env`, `work/`, `results/raw/*.jsonl`, `results/evaluations/*.csv`, and generated report files. Keep `.gitkeep` and tracked example fixtures explicitly unignored.

- [ ] **Step 4: Implement the exact enums and schema invariants**

In `enums.py`, define string enums for the six categories, three difficulties, six answer types, eight error categories, five run statuses, and three rubric scales. In `schemas.py`, implement the tested models with `ConfigDict(extra="forbid")`, UTC timestamps, non-empty strings, and a model validator enforcing paired figure fields and numeric tolerances.

Use these enum values exactly:

```python
class Category(StrEnum):
    ALGEBRA = "algebra"
    CALCULUS = "calculus"
    PROBABILITY = "probability"
    LOGIC = "logic"
    GRAPH_INTERPRETATION = "graph_interpretation"
    WORD_PROBLEMS = "word_problems"


class Difficulty(StrEnum):
    FOUNDATIONAL = "foundational"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class AnswerType(StrEnum):
    EXACT = "exact"
    NUMERIC = "numeric"
    RATIONAL = "rational"
    SYMBOLIC = "symbolic"
    SET_INTERVAL = "set_interval"
    STRUCTURED_TEXT = "structured_text"
```

Export package version `__version__ = "0.1.0"`. Make `__main__.py` import `app` from `cli` and invoke it; create a minimal `cli.py` with an empty Typer app so the module is importable.

- [ ] **Step 5: Install the package locally and run the schema tests**

Run: `python -m pip install -e ".[dev]"`

Run: `python -m pytest tests/unit/test_schemas.py -v`

Expected: all schema tests PASS.

- [ ] **Step 6: Run static checks and commit**

Run: `python -m ruff check src tests`

Run: `python -m mypy src`

Expected: both commands exit 0.

Commit:

```text
git add pyproject.toml .gitignore src/math_benchmark tests/unit/test_schemas.py
git commit -m "build: establish typed benchmark package"
```

---

### Task 2: Dataset Loading and Integrity Validation

**Files:**
- Create: `src/math_benchmark/dataset.py`
- Modify: `src/math_benchmark/cli.py`
- Create: `tests/unit/test_dataset.py`
- Create: `tests/unit/test_cli_validate.py`
- Create: `tests/fixtures/problems/valid_twenty.jsonl`
- Create: `tests/fixtures/problems/chart.png`
- Create: `tests/fixtures/problems/chart.csv`

**Interfaces:**
- Consumes: `Problem`, `Category`, and `Difficulty` from Task 1.
- Produces: `EXPECTED_CATEGORY_COUNTS`, `EXPECTED_DIFFICULTY_COUNTS`, `DatasetValidationError`, `load_problems(path: Path) -> list[Problem]`, `validate_benchmark(problems: Sequence[Problem], data_root: Path, enforce_distribution: bool = True) -> None`, and CLI command `validate`.

- [ ] **Step 1: Write failing loader and integrity tests**

Create tests for valid JSONL, malformed JSON with a line number, duplicate IDs, incorrect distribution, missing files, absolute figure paths, and `..` path traversal:

```python
def test_duplicate_problem_ids_are_rejected(tmp_path: Path) -> None:
    item = valid_problem_dict()
    path = tmp_path / "problems.jsonl"
    path.write_text("\n".join([json.dumps(item), json.dumps(item)]), encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="duplicate problem_id ALG-F-001"):
        validate_benchmark(load_problems(path), tmp_path, enforce_distribution=False)


@pytest.mark.parametrize("bad_path", ["../secret.png", "C:/secret.png"])
def test_figure_paths_cannot_escape_data_root(tmp_path: Path, bad_path: str) -> None:
    problem = valid_problem(figure_path=bad_path, figure_source_path="chart.csv")
    with pytest.raises(DatasetValidationError, match="must remain inside data root"):
        validate_benchmark([problem], tmp_path, enforce_distribution=False)


def test_exact_distribution_is_required(valid_twenty: list[Problem], tmp_path: Path) -> None:
    validate_benchmark(valid_twenty, tmp_path)
    with pytest.raises(DatasetValidationError, match="category distribution"):
        validate_benchmark(valid_twenty[:-1], tmp_path)
```

In `tests/unit/test_cli_validate.py`, invoke Typer's `CliRunner` with the valid fixture and assert exit code 0 plus `20 problems valid`. Invoke it with a malformed two-line JSONL file and assert a nonzero exit plus the exact failing line number.

Build `valid_twenty.jsonl` from short synthetic fixture records whose questions begin with `TEST FIXTURE — NOT BENCHMARK DATA`. It must satisfy the exact 20-item matrix without pretending to be real results.

- [ ] **Step 2: Run the tests and verify missing symbols**

Run: `python -m pytest tests/unit/test_dataset.py -v`

Expected: FAIL because `dataset.py` does not exist.

- [ ] **Step 3: Implement JSONL loading with actionable errors**

Implement UTF-8 loading one line at a time. Blank lines are ignored. Wrap JSON and Pydantic failures in `DatasetValidationError` containing the path and one-based line number. Reject an empty file.

```python
def load_problems(path: Path) -> list[Problem]:
    problems: list[Problem] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        try:
            problems.append(Problem.model_validate_json(raw_line))
        except (ValueError, ValidationError) as exc:
            raise DatasetValidationError(f"{path}:{line_number}: {exc}") from exc
    if not problems:
        raise DatasetValidationError(f"{path}: dataset is empty")
    return problems
```

- [ ] **Step 4: Implement safe path and distribution checks**

Use `Path.resolve()` plus `Path.is_relative_to()` to guarantee both figure files remain under `data_root`. Check duplicate IDs before checking counts. Use these constants:

```python
EXPECTED_CATEGORY_COUNTS = {
    Category.ALGEBRA: 4,
    Category.CALCULUS: 4,
    Category.PROBABILITY: 3,
    Category.LOGIC: 3,
    Category.GRAPH_INTERPRETATION: 3,
    Category.WORD_PROBLEMS: 3,
}
EXPECTED_DIFFICULTY_COUNTS = {
    Difficulty.FOUNDATIONAL: 5,
    Difficulty.INTERMEDIATE: 9,
    Difficulty.ADVANCED: 6,
}
```

- [ ] **Step 5: Run targeted and full tests**

Implement `validate` in `cli.py` with `--dataset` defaulting to `data/problems.jsonl` and `--data-root` defaulting to `data`. It must print the benchmark version and category/difficulty counts only after all checks pass.

Run: `python -m pytest tests/unit/test_dataset.py tests/unit/test_cli_validate.py -v`

Run: `python -m pytest tests/unit/test_schemas.py tests/unit/test_dataset.py -v`

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```text
git add src/math_benchmark/dataset.py src/math_benchmark/cli.py tests/unit/test_dataset.py tests/unit/test_cli_validate.py tests/fixtures/problems
git commit -m "feat: validate benchmark dataset integrity"
```

---

### Task 3: Answer Extraction and Primitive Validators

**Files:**
- Create: `src/math_benchmark/validators/__init__.py`
- Create: `src/math_benchmark/validators/models.py`
- Create: `src/math_benchmark/validators/extraction.py`
- Create: `src/math_benchmark/validators/exact.py`
- Create: `src/math_benchmark/validators/numeric.py`
- Create: `src/math_benchmark/validators/registry.py`
- Create: `tests/unit/validators/test_extraction.py`
- Create: `tests/unit/validators/test_primitive.py`

**Interfaces:**
- Consumes: `Problem`, `AnswerType`, and `ValidationSpec` from Task 1.
- Produces: `ExtractedAnswer(text: str | None, needs_review: bool, reason: str | None)`, `ValidationResult(is_correct: bool | None, needs_review: bool, normalized_actual: str | None, reason: str)`, `extract_final_answer(response: str) -> ExtractedAnswer`, and `validate_answer(problem: Problem, extracted: ExtractedAnswer) -> ValidationResult`.

- [ ] **Step 1: Write failing extraction tests**

Pin these decisions:

```python
def test_uses_single_required_marker() -> None:
    result = extract_final_answer("Work.\nFinal answer: 6")
    assert result == ExtractedAnswer(text="6", needs_review=False, reason=None)


def test_missing_marker_proposes_last_line_for_review() -> None:
    result = extract_final_answer("Work.\nTherefore x = 6")
    assert result.text == "Therefore x = 6"
    assert result.needs_review is True
    assert result.reason == "required marker missing"


def test_conflicting_repeated_markers_require_review() -> None:
    result = extract_final_answer("Final answer: 5\nCorrection.\nFinal answer: 6")
    assert result.text == "6"
    assert result.needs_review is True
    assert result.reason == "conflicting final-answer markers"


def test_empty_response_has_no_extractable_answer() -> None:
    assert extract_final_answer("   ").text is None
```

- [ ] **Step 2: Write failing primitive validator tests**

Include exact normalization, Unicode minus, rational equivalence, decimal tolerances, invalid numbers, and non-finite values:

```python
@pytest.mark.parametrize("actual", ["6", "6.0", "12/2", "+6"])
def test_rational_equivalence(actual: str, rational_problem: Problem) -> None:
    result = validate_answer(rational_problem, ExtractedAnswer(actual, False, None))
    assert result.is_correct is True


def test_numeric_tolerance_is_explicit(decimal_problem: Problem) -> None:
    assert validate_answer(decimal_problem, ExtractedAnswer("0.3334", False, None)).is_correct
    assert not validate_answer(decimal_problem, ExtractedAnswer("0.34", False, None)).is_correct


@pytest.mark.parametrize("actual", ["nan", "inf", "-inf", "six"])
def test_invalid_numeric_answers_require_review(actual: str, decimal_problem: Problem) -> None:
    result = validate_answer(decimal_problem, ExtractedAnswer(actual, False, None))
    assert result.is_correct is None
    assert result.needs_review is True
```

- [ ] **Step 3: Run tests to verify failures**

Run: `python -m pytest tests/unit/validators/test_extraction.py tests/unit/validators/test_primitive.py -v`

Expected: FAIL because validator modules do not exist.

- [ ] **Step 4: Implement extraction and immutable result models**

Use frozen dataclasses. Normalize CRLF, trim whitespace, and match `Final answer:` case-insensitively only at line starts. When repeated markers normalize to the same text, accept the last without review; when they differ, keep the last and require review.

- [ ] **Step 5: Implement exact, rational, and numeric validation**

Exact normalization may standardize whitespace, Unicode minus, surrounding dollar signs, and terminal punctuation. Rational parsing must use `fractions.Fraction`, not `eval`. Numeric parsing must use `Decimal` and reject non-finite values.

```python
def within_tolerance(actual: Decimal, expected: Decimal, absolute: Decimal, relative: Decimal) -> bool:
    difference = abs(actual - expected)
    allowed = max(absolute, relative * abs(expected))
    return difference <= allowed
```

Register validators by `AnswerType`; unsupported types return a review-required result rather than raising from user data.

- [ ] **Step 6: Run the validator tests and all existing tests**

Run: `python -m pytest tests/unit/validators -v`

Run: `python -m pytest tests/unit -v`

Expected: all tests PASS.

- [ ] **Step 7: Commit**

```text
git add src/math_benchmark/validators tests/unit/validators
git commit -m "feat: extract and validate primitive answers"
```

---

### Task 4: Bounded Symbolic, Set, and Structured Validators

**Files:**
- Create: `src/math_benchmark/validators/bounded.py`
- Create: `src/math_benchmark/validators/symbolic.py`
- Create: `src/math_benchmark/validators/set_interval.py`
- Create: `src/math_benchmark/validators/structured.py`
- Modify: `src/math_benchmark/validators/registry.py`
- Create: `tests/unit/validators/test_symbolic.py`
- Create: `tests/unit/validators/test_set_interval.py`
- Create: `tests/unit/validators/test_structured.py`

**Interfaces:**
- Consumes: `ValidationResult` and registry hook from Task 3.
- Produces: `BoundedExecutionResult`, `run_bounded(callable_path: str, args: Sequence[object], timeout_seconds: float) -> BoundedExecutionResult`, `symbolic_equivalent`, `set_interval_equivalent`, and `structured_text_equivalent` registered through `validate_answer`.

- [ ] **Step 1: Write failing safety and equivalence tests**

```python
@pytest.mark.parametrize("unsafe", [
    "__import__('os').system('whoami')",
    "open('secret.txt').read()",
    "x.__class__",
])
def test_symbolic_payloads_outside_allowlist_require_review(unsafe: str, symbolic_problem: Problem) -> None:
    result = validate_answer(symbolic_problem, ExtractedAnswer(unsafe, False, None))
    assert result.is_correct is None
    assert result.needs_review is True
    assert "unsupported symbolic syntax" in result.reason


def test_symbolic_equivalence() -> None:
    assert symbolic_equivalent("(x + 1)**2", "x**2 + 2*x + 1", {}, 2.0).is_correct


def test_symbolic_input_length_is_bounded(symbolic_problem: Problem) -> None:
    result = validate_answer(symbolic_problem, ExtractedAnswer("x+" * 300, False, None))
    assert result.needs_review is True
    assert result.reason == "symbolic input exceeds 512 characters"


def test_worker_timeout_returns_review(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "math_benchmark.validators.symbolic.run_bounded",
        lambda *args, **kwargs: BoundedExecutionResult.timed_out(),
    )
    result = symbolic_equivalent("x", "x", {}, 0.01)
    assert result.needs_review is True
    assert result.reason == "symbolic validation timed out"
```

Add interval tests for `(-inf, 2] U [5, inf)`, unordered finite sets, open/closed endpoint differences, and malformed notation. Add structured-text tests that compare a canonical JSON object with keys such as `valid`, `conclusion`, and `assignments` while ignoring object-key order but not values.

- [ ] **Step 2: Run tests to verify failures**

Run: `python -m pytest tests/unit/validators/test_symbolic.py tests/unit/validators/test_set_interval.py tests/unit/validators/test_structured.py -v`

Expected: FAIL because the new validators do not exist.

- [ ] **Step 3: Implement a process-bounded worker**

Use `multiprocessing.get_context("spawn")`, a queue, and a child process. On timeout, terminate and join the child. Accept only import paths rooted under `math_benchmark.validators`; reject arbitrary callables. Return a typed timeout result rather than leaving a child alive.

- [ ] **Step 4: Implement restricted symbolic equivalence**

Allow only identifiers listed in the problem validation options plus `sin`, `cos`, `exp`, `log`, `sqrt`, `pi`, and `E`. Reject attributes, brackets, quotes, underscores, and source strings longer than 512 characters before parsing. Parse with SymPy using an explicit local dictionary and empty global transformations, then check `simplify(actual - expected) == 0` in the bounded child.

- [ ] **Step 5: Implement set/interval and structured comparisons**

Parse interval unions with a small grammar supporting parentheses, brackets, finite Decimal endpoints, `inf`, `-inf`, commas, and `U`. Canonicalize into sorted disjoint components. Parse structured answers as JSON only and recursively compare required keys and primitive values; malformed text requires review.

- [ ] **Step 6: Register validators and run tests**

Run: `python -m pytest tests/unit/validators -v`

Run: `python -m ruff check src tests`

Expected: all tests PASS and Ruff exits 0.

- [ ] **Step 7: Commit**

```text
git add src/math_benchmark/validators tests/unit/validators
git commit -m "feat: add bounded advanced answer validators"
```

---

### Task 5: Author and Verify Algebra and Calculus Problems

**Files:**
- Create: `data/problem_sources/algebra.jsonl`
- Create: `data/problem_sources/calculus.jsonl`
- Create: `tests/content/test_algebra_calculus.py`

**Interfaces:**
- Consumes: `Problem`, `load_problems`, and `validate_answer`.
- Produces: eight fully reviewed source records with stable IDs and references used by Task 8.

- [ ] **Step 1: Lock the eight-item content manifest in a test**

Use this exact manifest and assert ID, difficulty, answer type, and canonical answer:

| ID | Difficulty | Exact problem and expected result |
|---|---|---|
| `ALG-F-001` | foundational | Solve `(3/4)(x - 2) + (1/3)(x + 6) = 7`; answer `x = 6`. |
| `ALG-I-002` | intermediate | Intersections of `y=x^2-4x+3` and `y=2x-1`; answer `{(3-sqrt(5), 5-2sqrt(5)), (3+sqrt(5), 5+2sqrt(5))}`. |
| `ALG-I-003` | intermediate | Real `a` for which roots of `x^2-(a+1)x+a=0` differ by 3; answer `{-2, 4}`. |
| `ALG-A-004` | advanced | Degree-at-most-3 polynomial with `p(x)-p(x-1)=3x^2-3x+1` and `p(0)=2`; answer `p(x)=x^3+2`. |
| `CAL-F-001` | foundational | For `f(x)=x^3-3x^2-9x+5`, find increasing/decreasing intervals and local extrema; answer increasing `(-inf,-1) U (3,inf)`, decreasing `(-1,3)`, local max `(-1,10)`, local min `(3,-22)`. |
| `CAL-I-002` | intermediate | Evaluate `integral_0^1 x exp(x^2) dx`; answer `(e-1)/2`. |
| `CAL-I-003` | intermediate | Evaluate `lim_{x->0}(exp(2x)-1-2x)/x^2`; answer `2`. |
| `CAL-A-004` | advanced | Largest rectangle symmetric about the y-axis under `y=12-x^2` and above the x-axis; answer half-width `2`, height `8`, maximum area `32`. |

The prompts must spell out domains and requested forms. `ALG-I-002`, `ALG-I-003`, and `CAL-F-001` use `set_interval` or `structured_text` ground truths rather than fragile display-string comparison.

- [ ] **Step 2: Run the manifest test and verify missing data**

Run: `python -m pytest tests/content/test_algebra_calculus.py -v`

Expected: FAIL because the two source files do not exist.

- [ ] **Step 3: Write the algebra records with complete solution checkpoints**

Each record must include a concise English question, canonical ground truth, a derivation that reaches the listed answer, tool-assisted verification notes, and `source_type="authored"`. Required checks are substitution for `ALG-F-001`, solving the simultaneous equations for `ALG-I-002`, factorization `(x-1)(x-a)` for `ALG-I-003`, and the finite-difference identity `x^3-(x-1)^3` for `ALG-A-004`.

- [ ] **Step 4: Run algebra-only content tests**

Run: `python -m pytest tests/content/test_algebra_calculus.py -k algebra -v`

Expected: algebra tests PASS.

- [ ] **Step 5: Write the calculus records with complete solution checkpoints**

Required checks are a derivative sign chart for `CAL-F-001`, substitution `u=x^2` for `CAL-I-002`, a second-order expansion or two justified L'Hôpital applications for `CAL-I-003`, and derivative plus endpoint/global analysis for `CAL-A-004`.

- [ ] **Step 6: Run all eight content tests**

Run: `python -m pytest tests/content/test_algebra_calculus.py -v`

Expected: all eight records parse, satisfy their category counts, and accept their own ground truths.

- [ ] **Step 7: Commit**

```text
git add data/problem_sources/algebra.jsonl data/problem_sources/calculus.jsonl tests/content/test_algebra_calculus.py
git commit -m "content: add verified algebra and calculus problems"
```

---

### Task 6: Author and Verify Probability, Logic, and Word Problems

**Files:**
- Create: `data/problem_sources/probability.jsonl`
- Create: `data/problem_sources/logic.jsonl`
- Create: `data/problem_sources/word_problems.jsonl`
- Create: `tests/content/test_probability_logic_words.py`

**Interfaces:**
- Consumes: the same data and validator interfaces as Task 5.
- Produces: nine reviewed source records used by Task 8.

- [ ] **Step 1: Lock the nine-item manifest in tests**

| ID | Difficulty | Exact problem and expected result |
|---|---|---|
| `PROB-F-001` | foundational | Disease prevalence 0.10, sensitivity 0.90, specificity 0.80; given a positive test, probability of disease is `1/3`. |
| `PROB-I-002` | intermediate | Choose box A with probability 0.40 and B with 0.60; A has 3 red/2 blue, B has 1 red/4 blue; given red, probability A is `2/3`. |
| `PROB-A-003` | advanced | Expected fair-coin tosses until pattern HHT first occurs; answer `8`, derived with explicit states. |
| `LOG-F-001` | foundational | Determine validity of `p -> q`, `not q`, therefore `not p`; answer valid by modus tollens. |
| `LOG-I-002` | intermediate | Negate the epsilon definition `for all epsilon>0 exists N for all n>=N, |a_n-L|<epsilon`; exact negation has `exists epsilon>0 for all N exists n>=N, |a_n-L|>=epsilon`. |
| `LOG-A-003` | advanced | Knights/knaves: A says “B is a knave”; B says “A and C are the same type”; C says “A is a knight”; unique answer A knave, B knight, C knave. |
| `WORD-I-001` | intermediate | Add 50% solution to 30 L of 20% solution to reach 35%; answer `30 L`. |
| `WORD-I-002` | intermediate | Pipe A fills in 6 h, B in 4 h, drain empties in 12 h; A and drain start, B joins after 1 h; total fill time `15/4 h` or `3 h 45 min`. |
| `WORD-A-003` | advanced | 100 L well-mixed tank starts with 10 kg salt; pure water enters and mixture exits at 5 L/min; time to 2 kg is `20 ln(5)` minutes, approximately `32.19`. |

- [ ] **Step 2: Run the content test and verify missing files**

Run: `python -m pytest tests/content/test_probability_logic_words.py -v`

Expected: FAIL because the source files do not exist.

- [ ] **Step 3: Write probability records and independent checks**

Use exact rational arithmetic for both Bayes problems. For HHT, define states for matched prefixes `""`, `H`, and `HH`, derive the linear expectation equations, and solve them to 8. Do not cite the result as a memorized pattern formula.

- [ ] **Step 4: Write logic records and truth checks**

Represent the first item as `{ "valid": true, "rule": "modus tollens" }`, the quantifier answer as a normalized structured object, and the knights/knaves answer as `{ "A": "knave", "B": "knight", "C": "knave" }`. Enumerate all eight truth assignments in the verification notes for the advanced item.

- [ ] **Step 5: Write word-problem records and dimensional checks**

Show concentration conservation, net rates, and the differential equation `Q'(t)=-(1/20)Q(t)` respectively. Include units in both display answers and instructions while keeping unit-free canonical numeric values for validators.

- [ ] **Step 6: Run tests and commit**

Run: `python -m pytest tests/content/test_probability_logic_words.py -v`

Expected: all nine records PASS their schema and reference-answer checks.

```text
git add data/problem_sources/probability.jsonl data/problem_sources/logic.jsonl data/problem_sources/word_problems.jsonl tests/content/test_probability_logic_words.py
git commit -m "content: add probability logic and word problems"
```

---

### Task 7: Generate and Verify Three Graph Problems

**Files:**
- Create: `data/figure_source_data/GRF-F-001.csv`
- Create: `data/figure_source_data/GRF-I-002.csv`
- Create: `data/figure_source_data/GRF-A-003.csv`
- Create: `scripts/generate_figures.py`
- Create: `data/figures/GRF-F-001.png`
- Create: `data/figures/GRF-I-002.png`
- Create: `data/figures/GRF-A-003.png`
- Create: `data/problem_sources/graph_interpretation.jsonl`
- Create: `tests/content/test_graph_problems.py`

**Interfaces:**
- Consumes: dataset schema and validator interfaces.
- Produces: `generate_all(source_dir: Path, output_dir: Path) -> list[Path]` and three complete graph records used by Task 8.

- [ ] **Step 1: Write failing source-data and reproducibility tests**

Lock these exact datasets and expected answers:

```python
EXPECTED = {
    "GRF-F-001": {
        "rows": [("Jan", 12), ("Feb", 15), ("Mar", 15), ("Apr", 21), ("May", 18)],
        "answer": {"largest_increase": 6, "from": "Mar", "to": "Apr"},
    },
    "GRF-I-002": {
        "rows": [(0, 0), (2, 6), (5, 12), (6, 12)],
        "answer": {"fastest_interval": [0, 2], "speed": 3, "overall_average_speed": 2},
    },
    "GRF-A-003": {
        "rows": [(0, -2), (2, 0), (4, 2), (6, 0)],
        "answer": {"minimum_x": 2, "net_change_0_to_6": 2},
    },
}
```

Test that `generate_all` creates exactly three PNG files, each at least 800×500 pixels, and that running it twice produces the same pixel dimensions and file hashes.

- [ ] **Step 2: Run tests to verify missing assets**

Run: `python -m pytest tests/content/test_graph_problems.py -v`

Expected: FAIL because sources and generator do not exist.

- [ ] **Step 3: Create source CSV files and deterministic plotting code**

Use Matplotlib's noninteractive `Agg` backend, a fixed style, explicit 120 DPI, fixed dimensions, accessible high-contrast colors, labeled axes, units, legends only when needed, and `metadata={"Software": "llm-math-reasoning-benchmark"}`. Do not encode the answer in filenames, captions, or annotations.

- [ ] **Step 4: Generate the three committed PNG files**

Run: `python scripts/generate_figures.py`

Expected: exactly three paths are printed and the script exits 0.

- [ ] **Step 5: Write graph-problem records**

- `GRF-F-001` asks for the largest month-to-month increase and its endpoints; difficulty foundational.
- `GRF-I-002` asks for the fastest interval and overall average speed from a distance-time plot; difficulty intermediate.
- `GRF-A-003` states that the plot is `f'(x)` and asks where `f` reaches its minimum on `[0,6]` and for `f(6)-f(0)`; difficulty advanced.

All three use structured ground truths and point to both the PNG and CSV source.

- [ ] **Step 6: Run tests and inspect images**

Run: `python -m pytest tests/content/test_graph_problems.py -v`

Expected: all tests PASS. Open each PNG once and verify that labels are legible and no point or line is clipped.

- [ ] **Step 7: Commit**

```text
git add data/figure_source_data data/figures data/problem_sources/graph_interpretation.jsonl scripts/generate_figures.py tests/content/test_graph_problems.py
git commit -m "content: add reproducible graph interpretation problems"
```

---

### Task 8: Build and Freeze the Canonical 20-Problem Dataset

**Files:**
- Create: `scripts/build_dataset.py`
- Create: `data/problems.jsonl`
- Create: `data/CHANGELOG.md`
- Create: `tests/content/test_canonical_dataset.py`

**Interfaces:**
- Consumes: all six category source files, `load_problems`, `validate_benchmark`, and `validate_answer`.
- Produces: `build_dataset(source_dir: Path, output_path: Path) -> list[Problem]` and canonical benchmark version `1.0.0`.

- [ ] **Step 1: Write failing canonical-dataset tests**

Test exact count/distribution, unique IDs, English non-empty copy, every ground truth accepted by its validator, all figure hashes stable, all `source_type` values valid, and deterministic build bytes:

```python
def test_every_ground_truth_is_accepted(canonical_problems: list[Problem]) -> None:
    for problem in canonical_problems:
        extracted = ExtractedAnswer(problem.ground_truth.canonical, False, None)
        result = validate_answer(problem, extracted)
        assert result.is_correct is True, f"{problem.problem_id}: {result.reason}"


def test_build_is_byte_for_byte_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    build_dataset(PROBLEM_SOURCE_DIR, first)
    build_dataset(PROBLEM_SOURCE_DIR, second)
    assert first.read_bytes() == second.read_bytes()
```

- [ ] **Step 2: Run the test and verify the canonical file is missing**

Run: `python -m pytest tests/content/test_canonical_dataset.py -v`

Expected: FAIL because `build_dataset.py` and `data/problems.jsonl` do not exist.

- [ ] **Step 3: Implement deterministic merging and freezing**

Load source files in the fixed order algebra, calculus, probability, logic, graph interpretation, word problems. Sort within each file by `problem_id`, validate without distribution enforcement, merge, apply full distribution validation, and write compact UTF-8 JSON with sorted keys and one terminal newline. Refuse to overwrite a canonical file with a different benchmark version unless `--new-version` is supplied.

- [ ] **Step 4: Build the canonical dataset**

Run: `python scripts/build_dataset.py`

Expected: reports 20 valid problems at benchmark version `1.0.0` and exits 0.

- [ ] **Step 5: Add the benchmark changelog**

Create `data/CHANGELOG.md` with version `1.0.0`, the freeze date, exact counts, and the statement that pilot outputs are excluded from the comparison. Document that future wording or ground-truth corrections require a version increment.

- [ ] **Step 6: Run content and full unit tests**

Run: `python -m pytest tests/content tests/unit -v`

Expected: all tests PASS.

- [ ] **Step 7: Commit**

```text
git add scripts/build_dataset.py data/problems.jsonl data/CHANGELOG.md tests/content/test_canonical_dataset.py
git commit -m "content: freeze benchmark dataset version 1.0.0"
```

---

### Task 9: Provider Contract, Mock Provider, and Ollama Adapter

**Files:**
- Create: `configs/models.yaml`
- Create: `configs/prompts.yaml`
- Create: `src/math_benchmark/providers/__init__.py`
- Create: `src/math_benchmark/providers/base.py`
- Create: `src/math_benchmark/providers/mock.py`
- Create: `src/math_benchmark/providers/ollama.py`
- Create: `tests/unit/providers/test_mock.py`
- Create: `tests/unit/providers/test_ollama.py`
- Create: `tests/integration/test_ollama_live.py`

**Interfaces:**
- Consumes: `Problem` and run models from Task 1.
- Produces: `ModelConfig`, `PromptConfig`, `ModelIdentity`, `Provider` protocol with `resolve_model(model: ModelConfig) -> ModelIdentity` and `generate(problem: Problem, prompt: PromptConfig, model: ModelConfig) -> ProviderResponse`, `MockProvider`, and `OllamaProvider(base_url: str, client: httpx.Client)`.

- [ ] **Step 1: Write provider-contract and payload tests**

Use `httpx.MockTransport` and assert exact text-only and image payloads:

```python
def test_ollama_text_payload(mock_transport: MockTransport, text_problem: Problem) -> None:
    provider = OllamaProvider("http://localhost:11434", httpx.Client(transport=mock_transport))
    response = provider.generate(text_problem, prompt_config(), model_config("gemma3:4b"))
    assert response.raw_text == "Final answer: 6"
    sent = mock_transport.last_json
    assert sent["model"] == "gemma3:4b"
    assert sent["stream"] is False
    assert sent["options"] == {"temperature": 0, "num_ctx": 4096, "num_predict": 2048, "seed": 42}


def test_ollama_image_payload_contains_base64_png(
    mock_transport: MockTransport,
    graph_problem: Problem,
) -> None:
    provider = OllamaProvider("http://localhost:11434", httpx.Client(transport=mock_transport))
    provider.generate(graph_problem, prompt_config(), model_config("qwen2.5vl:3b"))
    images = mock_transport.last_json["messages"][0]["images"]
    assert len(images) == 1
    decoded = base64.b64decode(images[0], validate=True)
    assert decoded.startswith(b"\x89PNG\r\n\x1a\n")
```

Add a parameterized test whose cases map HTTP 500 to `ProviderUnavailableError`, invalid JSON and missing message content to `ProviderProtocolError`, and `httpx.ReadTimeout` to `ProviderTimeoutError`. Add `test_mock_provider_uses_problem_id_lookup` with two IDs and assert that each receives its configured response rather than positional output.

Add `test_resolve_model_records_ollama_digest`, mock `/api/show`, return digest `sha256:test-digest`, and assert `ModelIdentity(tag="gemma3:4b", digest="sha256:test-digest")`.

- [ ] **Step 2: Run provider tests to verify failures**

Run: `python -m pytest tests/unit/providers -v`

Expected: FAIL because providers do not exist.

- [ ] **Step 3: Create versioned configurations**

`configs/prompts.yaml` contains prompt version `1.0.0` and the exact two-line base instruction from the spec. `configs/models.yaml` contains both model tags, provider `ollama`, and two named profiles: `primary` with temperature 0, seed 42, and one repetition; `stability` with temperature 0.2, seeds `[1103, 2207, 3301]`, and three repetitions. Both profiles use `num_ctx=4096`, `num_predict=2048`, and a 300-second timeout. Do not use `latest` tags.

- [ ] **Step 4: Implement the provider protocol and mock**

Make response and error types explicit. The mock accepts a mapping from `problem_id` to raw response and returns fixed timestamps supplied by a clock dependency in tests.

- [ ] **Step 5: Implement the Ollama HTTP adapter**

POST to `/api/chat`, supply one user message containing the shared instruction plus question, and attach a base64 image only for graph problems. Implement `resolve_model` with `/api/show` and require a nonempty digest before a non-dry run begins. Convert HTTP and timeout errors into typed provider errors without retrying inside the adapter.

- [ ] **Step 6: Add the opt-in live integration test**

Mark it `@pytest.mark.integration`, skip unless `OLLAMA_INTEGRATION=1`, query the configured base URL, and run only `ALG-F-001` with one explicitly selected model. The default test suite must not attempt a network connection.

- [ ] **Step 7: Run tests and commit**

Run: `python -m pytest tests/unit/providers -v`

Run: `python -m pytest -m "not integration" -v`

Expected: all non-integration tests PASS.

```text
git add configs src/math_benchmark/providers tests/unit/providers tests/integration/test_ollama_live.py
git commit -m "feat: add reproducible model provider interface"
```

---

### Task 10: Append-Only Run Storage, Retry, Resume, and CLI Run

**Files:**
- Create: `src/math_benchmark/environment.py`
- Create: `src/math_benchmark/storage.py`
- Create: `src/math_benchmark/runner.py`
- Modify: `src/math_benchmark/cli.py`
- Create: `tests/unit/test_storage.py`
- Create: `tests/unit/test_environment.py`
- Create: `tests/unit/test_runner.py`
- Create: `tests/unit/test_cli_run.py`

**Interfaces:**
- Consumes: canonical problems, provider contract, configs, and `RunAttempt`.
- Produces: `collect_environment(command_runner: CommandRunner) -> EnvironmentSnapshot`, `JsonlAttemptStore(path: Path)`, `RunPlan`, `build_run_plan`, `run_benchmark`, and CLI command `run`.

- [ ] **Step 1: Write failing append-only storage tests**

```python
def test_append_never_rewrites_existing_attempt(tmp_path: Path) -> None:
    path = tmp_path / "attempts.jsonl"
    store = JsonlAttemptStore(path)
    store.append(attempt(run_id="run-1", raw_response="first"))
    before = path.read_bytes()
    store.append(attempt(run_id="run-2", raw_response="second"))
    assert path.read_bytes().startswith(before)
    assert [a.run_id for a in store.load_all()] == ["run-1", "run-2"]


def test_duplicate_run_id_is_rejected(tmp_path: Path) -> None:
    store = JsonlAttemptStore(tmp_path / "attempts.jsonl")
    store.append(attempt(run_id="same"))
    with pytest.raises(StorageError, match="duplicate run_id"):
        store.append(attempt(run_id="same"))
```

- [ ] **Step 2: Write failing runner edge-case tests**

Test one retry for timeout, no retry for invalid model content, successful items skipped on resume, failed items retried with linked IDs, benchmark-version mismatch rejected, prompt-version mismatch rejected, and a returned response without a final answer recorded as `success` rather than infrastructure failure.

```python
def test_resume_rejects_version_mismatch(existing_manifest: RunManifest) -> None:
    with pytest.raises(RunCompatibilityError, match="benchmark version"):
        build_run_plan(problems_v2(), configs(), existing_manifest=existing_manifest)


def test_resume_skips_success_and_retries_timeout() -> None:
    plan = build_run_plan(two_problems(), configs(), existing_manifest=partial_manifest())
    assert [(item.problem_id, item.attempt_number) for item in plan.items] == [("ALG-I-002", 2)]
```

In `test_environment.py`, inject fixed outputs for Python version, platform, CPU, total RAM, `nvidia-smi`, and Ollama version. Assert that missing `nvidia-smi` yields `gpu=None` rather than failing, while every collected field is serialized into the manifest.

- [ ] **Step 3: Run tests to verify failures**

Run: `python -m pytest tests/unit/test_storage.py tests/unit/test_runner.py tests/unit/test_cli_run.py -v`

Expected: FAIL because storage and runner do not exist.

- [ ] **Step 4: Implement environment capture, append-only JSONL, and atomic manifests**

Implement environment collection through injected command and system-info adapters so unit tests do not depend on the host. Capture Python, package, OS, CPU, total RAM, GPU when present, Ollama version, benchmark version, prompt version, exact model tag, and resolved model digest. Use line-buffered append plus `flush()` and `os.fsync()` for attempts. Write manifests to a sibling temporary file and replace atomically. Never reuse a `run_id`; use UUID4. Store `parent_run_id` for retries and SHA-256 hashes for prompt and image bytes.

- [ ] **Step 5: Implement run planning and retry policy**

Create one planned item for every model/problem pair. A provider timeout or provider error receives at most one retry. Completed success records are skipped on resume. `invalid_response` means the provider returned unusable protocol data; a mathematically empty but valid text response remains `success` for evaluation.

- [ ] **Step 6: Implement the CLI run command**

Support exact options:

```text
python -m math_benchmark run \
  --models qwen2.5vl:3b \
  --models gemma3:4b \
  --profile primary \
  --dataset data/problems.jsonl \
  --output results/raw/primary-20260923T180000Z.jsonl \
  --manifest results/raw/primary-20260923T180000Z.manifest.json
```

Also support `--dry-run`, which validates configuration and prints 40 planned attempts without contacting Ollama. Support the separate stability command with the five fixed IDs `ALG-I-003`, `CAL-A-004`, `PROB-A-003`, `LOG-I-002`, and `GRF-A-003`:

```text
python -m math_benchmark run \
  --models qwen2.5vl:3b \
  --models gemma3:4b \
  --profile stability \
  --problem-id ALG-I-003 \
  --problem-id CAL-A-004 \
  --problem-id PROB-A-003 \
  --problem-id LOG-I-002 \
  --problem-id GRF-A-003
```

The stability profile must plan exactly 30 attempts and write a manifest with `experiment_kind="stability"`; primary manifests use `experiment_kind="primary"`.

- [ ] **Step 7: Run focused and full tests**

Run: `python -m pytest tests/unit/test_environment.py tests/unit/test_storage.py tests/unit/test_runner.py tests/unit/test_cli_run.py -v`

Run: `python -m math_benchmark run --dry-run --models qwen2.5vl:3b --models gemma3:4b`

Expected: tests PASS and dry run prints exactly 40 planned attempts.

- [ ] **Step 8: Commit**

```text
git add src/math_benchmark/environment.py src/math_benchmark/storage.py src/math_benchmark/runner.py src/math_benchmark/cli.py tests/unit/test_environment.py tests/unit/test_storage.py tests/unit/test_runner.py tests/unit/test_cli_run.py
git commit -m "feat: run and resume immutable benchmark attempts"
```

---

### Task 11: Automated Evaluation and Human Annotation Workflow

**Files:**
- Create: `src/math_benchmark/evaluation.py`
- Modify: `src/math_benchmark/cli.py`
- Create: `docs/annotation_guide.md`
- Create: `tests/unit/test_evaluation.py`
- Create: `tests/fixtures/evaluations/attempts.jsonl`
- Create: `tests/fixtures/evaluations/completed.csv`

**Interfaces:**
- Consumes: raw attempts, problems, extraction, validators, `HumanEvaluation`, and error enums.
- Produces: `create_annotation_sheet`, `load_completed_evaluations`, `derive_fully_correct_solution`, `select_blinded_recheck(frame: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame`, and CLI command `evaluate`.

- [ ] **Step 1: Write failing rubric and consistency tests**

```python
def test_fully_correct_requires_all_four_conditions() -> None:
    evaluation = human_evaluation(
        final_answer_correct=True,
        correctness_score=4,
        reasoning_quality_score=3,
        instruction_following_score=2,
        primary_error="correct",
    )
    assert derive_fully_correct_solution(evaluation) is True
    assert derive_fully_correct_solution(evaluation.model_copy(update={"instruction_following_score": 1})) is False


def test_correct_label_conflicts_with_incorrect_answer() -> None:
    with pytest.raises(ValidationError, match="primary_error=correct"):
        human_evaluation(final_answer_correct=False, primary_error="correct")


def test_override_requires_reason_and_original_result() -> None:
    with pytest.raises(ValidationError, match="override_reason"):
        human_evaluation(human_override=True, override_reason=None)


@pytest.mark.parametrize("field,value", [
    ("correctness_score", 5),
    ("reasoning_quality_score", -1),
    ("instruction_following_score", 3),
])
def test_rubric_ranges_are_enforced(field: str, value: int) -> None:
    with pytest.raises(ValidationError):
        human_evaluation(**{field: value})
```

Also reject a non-correct label without `first_error_step`, a `correct` label with a secondary error, a completed evaluation with blank reviewer ID, and duplicate evaluation IDs.

Add a 40-row test fixture balanced across model and difficulty, call `select_blinded_recheck(frame, fraction=0.20, seed=20260923)`, and assert eight unique rows, both models represented, all three difficulties represented, and no score, error, or review-note columns exposed in the returned blinded sheet.

- [ ] **Step 2: Run tests and verify failures**

Run: `python -m pytest tests/unit/test_evaluation.py -v`

Expected: FAIL because evaluation functions do not exist or schema invariants are incomplete.

- [ ] **Step 3: Strengthen `HumanEvaluation` invariants**

Add fields for automatic result, human override, original automatic result, override reason, three rubric scores, primary/secondary error, first error step, confidence, review notes, reviewer ID, annotation timestamp, and blinded-pass flag. Enforce all tested cross-field constraints in one model validator.

- [ ] **Step 4: Implement annotation-sheet creation**

Join each successful raw response to its problem. Preserve the response verbatim. Add extracted answer, automated result, and blank human columns in a stable CSV order. Technical failures go to a separate technical-failures CSV and are not inserted into the human scoring sheet.

- [ ] **Step 5: Implement completed-evaluation loading and derived fields**

Parse with explicit dtypes, validate every row through `HumanEvaluation`, reject duplicates, and add `fully_correct_solution`. Do not coerce blank or out-of-range scores.

Implement deterministic stratified recheck selection across model and difficulty. The export contains only response and problem context plus a new blinded-review ID; it must omit original annotations and aggregate metrics.

- [ ] **Step 6: Write the annotation guide**

Document all rubric anchors verbatim from the spec, the earliest-causal-error rule, each error label with one correct and one incorrect example, confidence levels, override procedure, and the delayed blinded 20% re-annotation procedure. Explicitly state that rankings must remain hidden during annotation.

- [ ] **Step 7: Add and test the CLI evaluate command**

Run:

```text
python -m math_benchmark evaluate \
  --dataset data/problems.jsonl \
  --attempts tests/fixtures/evaluations/attempts.jsonl \
  --output work/plan-check/annotation_sheet.csv
```

Expected: the fixture produces the tested annotation rows and a separate technical-failure file.

Add `--create-recheck`, `--recheck-fraction 0.20`, and `--recheck-seed 20260923`; when selected, the command writes `blinded_recheck.csv` beside the main annotation sheet.

- [ ] **Step 8: Run tests and commit**

Run: `python -m pytest tests/unit/test_evaluation.py -v`

Run: `python -m pytest -m "not integration" -v`

Expected: all tests PASS.

```text
git add src/math_benchmark/schemas.py src/math_benchmark/evaluation.py src/math_benchmark/cli.py docs/annotation_guide.md tests/unit/test_evaluation.py tests/fixtures/evaluations
git commit -m "feat: add auditable human evaluation workflow"
```

---

### Task 12: Metrics, Confidence Intervals, Charts, and Report Rendering

**Files:**
- Create: `src/math_benchmark/analysis/__init__.py`
- Create: `src/math_benchmark/analysis/metrics.py`
- Create: `src/math_benchmark/analysis/plots.py`
- Create: `src/math_benchmark/analysis/report.py`
- Modify: `src/math_benchmark/cli.py`
- Create: `tests/unit/analysis/test_metrics.py`
- Create: `tests/unit/analysis/test_plots.py`
- Create: `tests/unit/analysis/test_report.py`
- Create: `tests/fixtures/analysis/evaluations.csv`

**Interfaces:**
- Consumes: completed evaluation CSV and run metadata.
- Produces: `overall_metrics`, `metrics_by_category`, `metrics_by_difficulty`, `error_distribution`, `annotation_agreement`, `wilson_interval`, `create_all_plots`, `render_report`, and CLI command `analyze`.

- [ ] **Step 1: Write hand-computed metric tests**

Use a four-row fixture with two models and assert exact values:

```python
def test_overall_metrics_are_hand_computed(evaluation_frame: pd.DataFrame) -> None:
    result = overall_metrics(evaluation_frame).set_index("model_name")
    assert result.loc["model-a", "n"] == 2
    assert result.loc["model-a", "final_answer_accuracy"] == pytest.approx(0.5)
    assert result.loc["model-a", "fully_correct_rate"] == pytest.approx(0.5)
    assert result.loc["model-a", "mean_correctness"] == pytest.approx(3.0)


def test_technical_failures_are_not_in_math_denominator() -> None:
    frame = fixture_with_one_timeout_and_two_scored_responses()
    result = overall_metrics(frame).iloc[0]
    assert result["n_scored"] == 2
    assert result["n_technical_failures"] == 1
```

Test the Wilson interval for 0/20, 10/20, and 20/20 against fixed expected values to four decimal places. Test that primary and stability-probe records cannot be combined by default.

Add paired original/recheck rows and assert `annotation_agreement` reports exact agreement and within-one-point agreement separately for correctness and reasoning, exact agreement for instruction following, and primary-error agreement. Assert that the function returns no Cohen's kappa field unless two distinct reviewer IDs are present.

- [ ] **Step 2: Write plot and report tests**

Assert exactly five named PNG outputs, nonzero dimensions, category item counts embedded in heatmap labels, and report headings for methods, results, case studies, stability, annotation quality, limitations, and reproduction. Assert that a fixture report begins with `SYNTHETIC TEST DATA — NOT EXPERIMENTAL RESULTS`.

- [ ] **Step 3: Run tests to verify failures**

Run: `python -m pytest tests/unit/analysis -v`

Expected: FAIL because analysis modules do not exist.

- [ ] **Step 4: Implement pure Pandas metric functions**

Every public function accepts and returns DataFrames without writing files. Use explicit denominators and carry `n` columns through all grouped tables. Implement the 95% Wilson interval directly with `z=1.959963984540054`; do not add SciPy.

Compute delayed recheck agreement from paired blinded-review IDs. If two distinct reviewers exist, include Cohen's kappa for categorical error labels; otherwise report intra-rater measures only and label them accordingly.

- [ ] **Step 5: Implement accessible plots from metric tables**

Use a colorblind-safe palette, start proportion axes at zero, cap accuracy axes at one, include item counts, and never plot a connected trend that implies continuous difficulty. Return created paths and close every figure.

- [ ] **Step 6: Implement report rendering**

Render Markdown from metric tables and plot paths. Refuse to render a final-looking report when all input records are marked fixture or synthetic unless `--fixture-report` is supplied; the fixture mode must add the warning banner tested above.

- [ ] **Step 7: Add and test the CLI analyze command**

Run:

```text
python -m math_benchmark analyze \
  --evaluations tests/fixtures/analysis/evaluations.csv \
  --output work/plan-check/report \
  --fixture-report
```

Expected: five PNG files, CSV metric tables, and one Markdown report are created.

- [ ] **Step 8: Run tests and commit**

Run: `python -m pytest tests/unit/analysis -v`

Run: `python -m pytest -m "not integration" -v`

Expected: all tests PASS.

```text
git add src/math_benchmark/analysis src/math_benchmark/cli.py tests/unit/analysis tests/fixtures/analysis
git commit -m "feat: analyze benchmark results reproducibly"
```

---

### Task 13: Documentation, Notebook, CI, and End-to-End Verification

**Files:**
- Create: `README.md`
- Create: `docs/methodology.md`
- Create: `docs/target-pc-runbook.md`
- Create: `notebooks/benchmark_analysis.ipynb`
- Create: `.github/workflows/ci.yml`
- Create: `tests/e2e/test_fixture_pipeline.py`
- Create: `tests/unit/test_documentation.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: every package and CLI interface from Tasks 1–12.
- Produces: a reviewer-facing repository, a thin reproducible notebook, network-free CI, and a complete Milestone A verification command.

- [ ] **Step 1: Write failing end-to-end and documentation tests**

The end-to-end test must use `MockProvider` to run two problems, write raw attempts, create an annotation sheet, load a committed completed fixture, and render a synthetic report in a temporary directory. Documentation tests must check that every README command exists in CLI help, every linked repository-relative path exists, and the README contains the phrases `20 problems`, `human evaluation`, `Ollama`, `limitations`, and `not experimental results`.

```python
def test_fixture_pipeline_runs_without_network(tmp_path: Path) -> None:
    attempts = run_fixture_benchmark(tmp_path, provider=MockProvider(FIXTURE_RESPONSES))
    sheet = create_annotation_sheet(PROBLEMS, attempts, tmp_path / "annotations.csv")
    report = render_fixture_report(COMPLETED_FIXTURE, tmp_path / "report")
    assert len(attempts) == 2
    assert sheet.exists()
    assert report.exists()
```

- [ ] **Step 2: Run tests to verify missing deliverables**

Run: `python -m pytest tests/e2e/test_fixture_pipeline.py tests/unit/test_documentation.py -v`

Expected: FAIL because documentation, notebook, and CI files do not exist.

- [ ] **Step 3: Write the README**

Use this section order: project summary, research questions, benchmark matrix, architecture, repository map, installation without Ollama, dataset validation, target-PC inference, human annotation, analysis, example outputs labeled synthetic, testing, limitations, roadmap, and license/provenance. Include exact commands from the spec and explain that real results are absent until the target-PC run.

- [ ] **Step 4: Write methodology and target-PC runbook**

`methodology.md` must document dataset freeze, prompts, parameters, metrics, Wilson intervals, stability separation, denominator rules, and all limitations. `target-pc-runbook.md` must include NVIDIA driver verification, official Ollama installation link, model pulls, `ollama list`, a text smoke test, an image smoke test, environment capture, 40-response primary run, annotation, blinded re-review, stability probe, report generation, and a final integrity check. It must not contain credentials or personal paths.

- [ ] **Step 5: Create a thin notebook**

Use `nbformat` to create a committed notebook whose code cells only load a selected evaluation CSV, call package metric functions, display tables, and call plot functions. The first Markdown cell must say that committed fixture output is synthetic and that real findings require Milestone B. Clear execution counts and outputs before committing.

- [ ] **Step 6: Add network-free GitHub Actions**

Create `.github/workflows/ci.yml` for pushes and pull requests on Python 3.11 and 3.12. Install `.[dev]`, run Ruff, mypy, `pytest -m "not integration"`, rebuild the dataset, and fail if `git diff --exit-code data/problems.jsonl data/figures` shows generated artifacts changed.

- [ ] **Step 7: Run complete Milestone A verification**

Run:

```text
python scripts/generate_figures.py
python scripts/build_dataset.py
python -m math_benchmark validate
python -m math_benchmark run --dry-run --models qwen2.5vl:3b --models gemma3:4b
python -m pytest -m "not integration" --cov=math_benchmark --cov-report=term-missing
python -m ruff check src tests scripts
python -m mypy src
git diff --exit-code data/problems.jsonl data/figures
```

Expected: every command exits 0; dry run reports 40 attempts; generated artifacts have no diff.

- [ ] **Step 8: Commit**

```text
git add README.md docs/methodology.md docs/target-pc-runbook.md notebooks .github tests/e2e tests/unit/test_documentation.py .gitignore
git commit -m "docs: complete reproducible benchmark portfolio"
```

---

### Task 14: Publish the Verified Repository to GitHub

**Files:**
- Modify: none unless the user requests a repository-description change before publication.

**Interfaces:**
- Consumes: a clean local repository with all Milestone A commits and authenticated GitHub CLI account `Buguerito`.
- Produces: public repository `https://github.com/Buguerito/llm-math-reasoning-benchmark` with remote `origin` and default branch `main`.

- [ ] **Step 1: Confirm publication settings at action time**

Show the user the exact owner, repository name, description, and `public` visibility:

```text
Owner: Buguerito
Repository: llm-math-reasoning-benchmark
Visibility: public
Description: A reproducible local-first benchmark for evaluating LLM mathematical reasoning.
```

Do not create the remote until the user confirms these exact settings.

- [ ] **Step 2: Verify the local branch and working tree**

Run:

```text
git status --short
git log --oneline --decorate -10
python -m pytest -m "not integration" -q
```

Expected: clean status, expected commit history, and all tests PASS.

- [ ] **Step 3: Rename the local default branch**

Run: `git branch -M main`

Expected: `git branch --show-current` prints `main`.

- [ ] **Step 4: Create and push the public repository**

Run:

```text
gh repo create Buguerito/llm-math-reasoning-benchmark \
  --public \
  --source . \
  --remote origin \
  --push \
  --description "A reproducible local-first benchmark for evaluating LLM mathematical reasoning."
```

Expected: GitHub prints the repository URL and the initial push succeeds.

- [ ] **Step 5: Verify the remote and CI**

Run:

```text
git remote -v
gh repo view Buguerito/llm-math-reasoning-benchmark --web=false
gh run list --repo Buguerito/llm-math-reasoning-benchmark --limit 5
```

Expected: `origin` points to the new repository, repository visibility is public, and the CI workflow appears. If CI is still running, capture and watch the newest run with these PowerShell commands:

```powershell
$workflowRunId = gh run list --repo Buguerito/llm-math-reasoning-benchmark --limit 1 --json databaseId --jq '.[0].databaseId'
gh run watch $workflowRunId --repo Buguerito/llm-math-reasoning-benchmark --exit-status
```

- [ ] **Step 6: Record publication outcome locally**

Add the repository URL to the README only if it is not already displayed automatically by GitHub, run the documentation test, and commit/push that one change if needed:

```text
git add README.md
git commit -m "docs: link published benchmark repository"
git push origin main
```

---

## Milestone B: Target-PC Experiment Handoff

Milestone B is deliberately not executed in the current environment. On the target PC, follow `docs/target-pc-runbook.md` in this order:

1. Install and verify Ollama and the NVIDIA driver.
2. Pull the two exact model tags and capture their digests.
3. Run one text and one image smoke item per model.
4. Execute the 40-response primary run at temperature zero.
5. Create and complete the human annotation sheet without viewing aggregate rankings.
6. Re-annotate a stratified 20% sample in a delayed blinded pass.
7. Execute the separate 30-response stability probe.
8. Generate the final tables, figures, and report from real evaluations.
9. Commit real results only after checking that no fixture is mixed into the primary data.

The final empirical commit message will be:

```text
results: add audited local-model benchmark findings
```
