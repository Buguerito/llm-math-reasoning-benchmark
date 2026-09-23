# Target-PC Runbook

This runbook is for the 16 GB RAM / NVIDIA RTX 2060 Super machine. It uses local models and requires no credentials.

## 1. Verify the NVIDIA driver

Open PowerShell and run:

```powershell
nvidia-smi
```

Confirm that the RTX 2060 Super appears and the driver is healthy. Ollama bundles its runtime; a separate CUDA toolkit is not required for this benchmark.

## 2. Install Python and Ollama

Install Python 3.11 or 3.12. Install Ollama from the [official Windows download page](https://ollama.com/download/windows), restart the terminal, then verify:

```powershell
ollama --version
```

## 3. Install the project

Clone the repository, enter it, and create an isolated environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## 4. Pull and inventory the models

```powershell
ollama pull qwen2.5vl:3b
ollama pull gemma3:4b
ollama list
```

Do not substitute a `latest` tag. Record the tags shown by `ollama list`; the benchmark manifest also requires Ollama to return a nonempty model digest.

## 5. Smoke-test text and image inputs

Text:

```powershell
ollama run gemma3:4b "Solve 2x+1=7 and end with Final answer: <answer>"
```

Image:

```powershell
ollama run qwen2.5vl:3b "Read ./data/figures/GRF-F-001.png and state the tallest month."
```

Confirm that each command returns a coherent response. These checks are pilots and must not enter the comparison.

## 6. Verify data and capture the plan

```powershell
python scripts/generate_figures.py
python scripts/build_dataset.py
python -m math_benchmark validate
python -m math_benchmark run --dry-run --models qwen2.5vl:3b --models gemma3:4b
```

The dry run must report 40 planned attempts. The real run automatically captures Python/packages, OS, CPU, total RAM, GPU when available, Ollama version, prompt/benchmark versions, model tags/digests, and input hashes.

## 7. Execute the 40-response primary run

Choose one timestamp and reuse it for both paths:

```powershell
python -m math_benchmark run --models qwen2.5vl:3b --models gemma3:4b --profile primary --dataset data/problems.jsonl --output results/raw/primary-TIMESTAMP.jsonl --manifest results/raw/primary-TIMESTAMP.manifest.json
```

Models run sequentially. If interrupted, repeat the same command and paths; successful pairs are skipped and eligible technical failures retain linked retry IDs.

## 8. Create and complete annotation

```powershell
python -m math_benchmark evaluate --dataset data/problems.jsonl --attempts results/raw/primary-TIMESTAMP.jsonl --output results/evaluations/annotation_sheet.csv
```

Follow [the annotation guide](annotation_guide.md). Keep model rankings hidden. Fill reviewer ID and timestamp, validate overrides, and version-control the completed sheet outside public raw fixtures if it contains unpublished study work.

## 9. Perform the delayed blinded 20% re-review

After a delay, export with `--create-recheck --recheck-fraction 0.20 --recheck-seed 20260923`. Complete the blinded sheet without viewing the first-pass scores, labels, notes, or aggregate rankings. Adjudicate discrepancies after both passes are frozen.

## 10. Run the stability probe

```powershell
python -m math_benchmark run --models qwen2.5vl:3b --models gemma3:4b --profile stability --problem-id ALG-I-003 --problem-id CAL-A-004 --problem-id PROB-A-003 --problem-id LOG-I-002 --problem-id GRF-A-003 --output results/raw/stability-TIMESTAMP.jsonl --manifest results/raw/stability-TIMESTAMP.manifest.json
```

This produces 30 planned attempts: five items, two models, and three fixed seeds. Never merge these records into the primary denominator.

## 11. Generate the report

```powershell
python -m math_benchmark analyze --evaluations results/evaluations/completed.csv --output results/reports/primary
```

Do not use `--fixture-report` for experimental data.

## 12. Final integrity check

```powershell
python -m pytest -m "not integration"
python -m ruff check src tests scripts
python -m mypy src
git diff --exit-code data/problems.jsonl data/figures
```

Archive the manifest, raw JSONL, completed annotations, agreement export, metric CSVs, plots, report, and Git commit SHA together.
