# LLM Mathematical Reasoning Evaluation Benchmark

## Project Summary

A local-first, reproducible portfolio project for evaluating mathematical reasoning in large language models. The frozen benchmark contains **20 problems** across six categories, deterministic validators, an auditable human evaluation workflow, and reproducible analysis. It runs free with local Ollama models; optional API providers can be added behind the same provider contract.

Committed example data are **synthetic test data, not experimental results**. Real model findings are intentionally absent until inference is performed on the target PC.

## Research Questions

1. How does final-answer accuracy differ by model, mathematical category, and difficulty?
2. How often is a correct answer supported by sound reasoning and complete instruction following?
3. Which causal failure modes occur most often?
4. How stable are selected responses across nonzero-temperature repetitions?

## Benchmark Matrix

| Category | Problems |
|---|---:|
| Algebra | 4 |
| Calculus | 4 |
| Probability | 3 |
| Logic | 3 |
| Graph interpretation | 3 |
| Word problems | 3 |

Difficulty totals are 5 foundational, 9 intermediate, and 6 advanced items. Three graph problems include frozen PNG assets generated from committed CSV data.

## Architecture

The workflow separates immutable inputs, model inference, automated validation, human annotation, and analysis. Raw attempts are append-only; changing an annotation never reruns inference. Exact, rational, numeric, symbolic, interval, and structured answers have conservative validators, with ambiguous cases routed to review.

## Repository Map

- [Canonical dataset](data/problems.jsonl) and [dataset changelog](data/CHANGELOG.md)
- [Model configuration](configs/models.yaml) and [prompt configuration](configs/prompts.yaml)
- [Package source](src/math_benchmark)
- [Annotation guide](docs/annotation_guide.md)
- [Methodology](docs/methodology.md)
- [Target-PC runbook](docs/target-pc-runbook.md)
- [Analysis notebook](notebooks/benchmark_analysis.ipynb)
- [Tests](tests)

## Installation Without Ollama

Dataset validation, tests, mock inference, evaluation, and synthetic analysis do not require Ollama:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

On macOS or Linux, activate with `source .venv/bin/activate`. No model is trained and no API key is required.

## Dataset Validation

Rebuild the portable JSONL artifact and validate the frozen distribution:

```powershell
python scripts/build_dataset.py
python -m math_benchmark validate
```

The committed PNGs are the canonical image inputs. `python scripts/generate_figures.py` is an authoring command for a new benchmark version, not a prerequisite for inference: Matplotlib raster bytes can vary across operating systems even when the chart content is unchanged.

## Target-PC Inference

Follow the [target-PC runbook](docs/target-pc-runbook.md) to install Ollama and pull `qwen2.5vl:3b` and `gemma3:4b`. Preview the primary plan without a network or running model:

```powershell
python -m math_benchmark run --dry-run --models qwen2.5vl:3b --models gemma3:4b
```

Remove `--dry-run` and provide `--output` plus `--manifest` on the target PC. The primary run produces 40 scored-response candidates: two models times 20 problems. Attempts are resumable, model digests are recorded, and timeouts/provider failures receive at most one linked retry.

## Human Annotation

Create the annotation sheet after inference:

```powershell
python -m math_benchmark evaluate --dataset data/problems.jsonl --attempts results/raw/primary.jsonl --output results/evaluations/annotation_sheet.csv
```

Reviewers score correctness, reasoning quality, and instruction following, then assign the earliest causal error. A delayed, blinded 20% recheck measures annotation consistency. Aggregate rankings stay hidden during annotation.

## Analysis

After completing and validating the annotation CSV:

```powershell
python -m math_benchmark analyze --evaluations results/evaluations/completed.csv --output results/reports/primary
```

The command writes metric tables, Wilson intervals, five accessible PNG charts, and a Markdown report. Primary and stability experiments are never combined by default.

## Example Outputs Labeled Synthetic

The committed fixtures exercise the complete pipeline but are synthetic and must not be presented as model performance. Generate their report with `--fixture-report`; the report begins with a warning banner.

## Testing

```powershell
python -m pytest -m "not integration"
python -m ruff check src tests scripts
python -m mypy src
```

The live Ollama test is opt-in through `OLLAMA_INTEGRATION=1`; CI never contacts Ollama.

## Limitations

- Twenty authored items support a transparent portfolio study, not broad claims about mathematical intelligence.
- Category-level samples are small and uncertainty intervals are correspondingly wide.
- Human scoring is subject to judgment; delayed rechecks quantify but do not eliminate this limitation.
- Local hardware, quantization, model digests, and Ollama versions can affect reproducibility.
- The initial release compares only two small vision-language models and contains no real target-PC results.

## Roadmap

- Run Milestone B on the 16 GB RAM / RTX 2060 Super target machine.
- Complete blinded annotation and stability probes.
- Add optional API providers without changing the dataset or scoring contract.
- Expand only under a new benchmark version with additional independently reviewed items.

## License and Provenance

Code is available under the [MIT License](LICENSE). Benchmark problems are authored for this project; provenance and transformations are recorded per item. Figures are generated from committed source data. See the [methodology](docs/methodology.md) before interpreting results.
