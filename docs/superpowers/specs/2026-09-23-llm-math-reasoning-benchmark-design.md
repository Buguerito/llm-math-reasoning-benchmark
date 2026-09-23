# LLM Mathematical Reasoning Evaluation Benchmark — Design Specification

**Date:** 2026-09-23  
**Status:** Approved conversational design  
**Audience:** Hiring teams for a Research Analyst – Advanced Math role  
**Deliverable language:** English

## 1. Purpose

This project will be a portfolio-quality, local-first benchmark for evaluating how well language models solve and explain mathematical reasoning problems. It is intended to demonstrate mathematical content creation, solution verification, research discipline, human evaluation, error analysis, Python engineering, Pandas analysis, and responsible interpretation of limited experimental evidence.

The first release is a rigorously reviewed 20-problem minimum viable benchmark. It must run for free with local models through Ollama and expose a provider interface that can support paid APIs later. No model training or fine-tuning is in scope.

The implementation will be developed without installing or running Ollama on the current computer. Unit tests will use clearly labeled fixtures and a mock provider. Real model inference will be performed later on the target Windows PC.

## 2. Goals and Non-Goals

### Goals

- Author and validate 20 original or clearly attributed mathematical reasoning problems in English.
- Cover algebra, calculus, probability, logic, graph interpretation, and word problems.
- Use progressive difficulty from foundational to advanced undergraduate-entry material.
- Preserve every raw model response and all inference metadata.
- Separate final-answer validation from qualitative reasoning evaluation.
- Apply an anchored human rubric for correctness, reasoning quality, and instruction following.
- Classify the first causal failure in incorrect solutions with a controlled error taxonomy.
- Analyze performance by model, category, and difficulty with Pandas.
- Produce a reproducible report with appropriate uncertainty and limitations.
- Run without a GPU for validation, testing, annotation, and analysis.

### Non-Goals

- Training or fine-tuning a language model.
- Claiming broad or statistically conclusive model rankings from 20 items.
- Using an LLM as the final authority for correctness or reasoning quality.
- Building a web application, hosted service, or interactive dashboard in the first release.
- Automatically generating the benchmark questions without human mathematical review.
- Including paid API results in the initial benchmark report.

## 3. Success Criteria

The project succeeds when a reviewer can:

1. Inspect the benchmark, reference solutions, provenance, and validation rules.
2. Run all schema, validator, analysis, and provider-contract tests without Ollama.
3. Install Ollama on another PC and execute both configured local models with documented commands.
4. Trace every reported score back to an immutable raw response and a human evaluation record.
5. Understand why each response was scored and, when incorrect, where the first material error occurred.
6. Reproduce all summary tables and charts from the saved evaluation data.
7. See limitations stated clearly enough to prevent overgeneralization.

## 4. Benchmark Composition

The 20 items will follow this fixed matrix:

| Category | Foundational | Intermediate | Advanced | Total |
|---|---:|---:|---:|---:|
| Algebra | 1 | 2 | 1 | 4 |
| Calculus | 1 | 2 | 1 | 4 |
| Probability | 1 | 1 | 1 | 3 |
| Logic | 1 | 1 | 1 | 3 |
| Graph Interpretation | 1 | 1 | 1 | 3 |
| Word Problems | 0 | 2 | 1 | 3 |
| **Total** | **5** | **9** | **6** | **20** |

Difficulty labels describe the reasoning burden within this benchmark, not universal educational standards:

- **Foundational:** one central concept, limited computation, and a short justification.
- **Intermediate:** multiple linked steps or a non-obvious modeling choice.
- **Advanced:** several dependent steps, a proof-like argument, or a subtle interpretation or assumption.

Each item must be self-contained, have one intended interpretation, state units and rounding rules when relevant, and require reasoning rather than formula recall alone. Items must not be superficial numerical variants of one template.

The three graph-interpretation items will use original images generated from committed source data. Both the image and its source table or generation parameters will be retained. The ground truth will be computed from the source data rather than estimated visually.

## 5. Canonical Problem Schema

Problems will be stored as one JSON object per line in `data/problems.jsonl`. Each object will contain:

| Field | Type | Requirement |
|---|---|---|
| `problem_id` | string | Unique, stable identifier |
| `category` | enum | One of the six approved categories |
| `subdomain` | string | More specific mathematical topic |
| `difficulty` | enum | `foundational`, `intermediate`, or `advanced` |
| `question` | string | Complete English prompt text |
| `answer_type` | enum | Exact, numeric, rational, symbolic, set/interval, or structured text |
| `ground_truth` | object | Display form, canonical form, and accepted equivalents |
| `reference_solution` | string | Reviewed step-by-step solution in English |
| `validation` | object | Validator name, tolerance, and type-specific options |
| `skills_tested` | string array | Concepts and reasoning operations assessed |
| `figure_path` | string or null | Relative path for a graph image |
| `figure_source_path` | string or null | Relative path to committed source data |
| `source` | object | Authored/adapted status and citation when applicable |
| `verification_notes` | string | Independent checks and ambiguity notes |
| `benchmark_version` | string | Dataset version containing the item |

The schema validator will reject unknown enum values, duplicate IDs, missing required fields, nonexistent figure paths, invalid tolerances, empty solutions, and a category/difficulty distribution that differs from the approved matrix.

## 6. Problem Authoring and Verification

Each problem will pass four gates before the dataset is frozen:

1. **Independent derivation:** recompute the answer without copying the draft solution.
2. **Tool-assisted check:** use SymPy or numerical calculation when the answer type permits it.
3. **Content audit:** check ambiguity, hidden assumptions, units, requested precision, and assigned difficulty.
4. **Pilot audit:** run preliminary responses only to identify defective wording or validation rules.

Pilot results will not be included in the final comparison. Any wording change after the pilot creates a new benchmark version. Once the final dataset is frozen, no item may be edited during the primary experiment. Corrections require a version increment and a documented change log.

Original problems will be labeled `authored`. Adapted problems must identify the source and describe the transformation. Time-sensitive facts and unverifiable external claims are prohibited.

## 7. System Architecture

The implementation will have five independent layers:

1. **Dataset layer:** loads, validates, and versions benchmark items.
2. **Provider layer:** sends a normalized request to a local or optional remote model.
3. **Validation layer:** extracts and checks final answers with type-specific deterministic validators.
4. **Evaluation layer:** stores human rubric scores, error labels, and adjudication notes.
5. **Analysis layer:** calculates metrics and creates tables, figures, and the final report.

The data flow is:

```text
Problems -> Dataset validation -> Model inference -> Raw response archive
         -> Final-answer validation -> Human evaluation -> Error classification
         -> Pandas analysis -> Tables, figures, and report
```

Inference, evaluation, and analysis must be rerunnable independently. Editing an annotation must not trigger model inference. Rebuilding charts must not modify raw responses.

### Provider Contract

All providers will implement one typed interface conceptually equivalent to:

```python
generate(problem, prompt_config, model_config) -> ModelResponse
```

`ModelResponse` will include the raw text, provider, exact model tag, model digest when available, parameters, latency, timestamps, completion status, image inputs, and technical error metadata. The Ollama implementation is required. The mock provider is required for tests. External API providers are optional extensions and must use environment variables for credentials.

### Command-Line Interface

The package will expose these commands:

```text
python -m math_benchmark validate
python -m math_benchmark run --models qwen2.5vl:3b gemma3:4b
python -m math_benchmark evaluate
python -m math_benchmark analyze
```

- `validate` checks the dataset, figures, solutions, and validator configuration.
- `run` executes selected models and writes immutable raw attempts plus a run manifest.
- `evaluate` applies deterministic validators and creates or updates the human annotation sheet.
- `analyze` reads completed evaluations and recreates all published outputs.

## 8. Repository Structure

```text
llm-math-reasoning-benchmark/
|-- README.md
|-- pyproject.toml
|-- configs/
|   |-- models.yaml
|   `-- prompts.yaml
|-- data/
|   |-- problems.jsonl
|   |-- figures/
|   `-- figure_source_data/
|-- src/math_benchmark/
|   |-- __init__.py
|   |-- __main__.py
|   |-- cli.py
|   |-- schemas.py
|   |-- dataset.py
|   |-- providers/
|   |-- validators/
|   |-- evaluation/
|   `-- analysis/
|-- results/
|   |-- raw/
|   |-- evaluations/
|   `-- reports/
|-- notebooks/
|   `-- benchmark_analysis.ipynb
|-- docs/
|   |-- methodology.md
|   `-- annotation_guide.md
`-- tests/
    |-- fixtures/
    |-- unit/
    `-- integration/
```

Reusable logic will live under `src/math_benchmark`. The notebook is a presentation and exploratory-analysis surface, not the source of truth for scoring logic.

## 9. Model Selection and Inference Protocol

The primary comparison will use:

- `qwen2.5vl:3b`, a vision-language model distributed through Ollama at approximately 3.2 GB.
- `gemma3:4b`, a vision-language model distributed through Ollama at approximately 3.3 GB.

Both accept text and image inputs, so all 20 items can be evaluated on the same sample. They will run one at a time on the target PC. The exact model tag and Ollama content digest must be saved because tags may later resolve to updated artifacts.

The shared base instruction is:

```text
Provide a concise solution that explains the key steps.
End your response with: Final answer: <answer>
```

Question-specific units, rounding rules, and output constraints remain in the problem text.

### Primary Run

- Two models and 20 problems, producing 40 scored responses.
- A new, stateless conversation for every problem.
- Identical prompt construction for both models.
- `temperature = 0`.
- `num_ctx = 4096`.
- `num_predict = 2048`.
- `seed = 42` when supported; support status is recorded.
- No examples, external tools, retrieval, or answer feedback.
- A five-minute per-attempt timeout.
- Complete model, environment, and run-manifest metadata.

### Stability Probe

A separate exploratory probe will select five problems stratified across difficulty and modality. Each selected problem will be run three times per model at `temperature = 0.2`, with distinct recorded seeds when supported. These 30 responses are analyzed separately and never mixed with primary-run accuracy.

### Optional Later Comparison

`qwen2-math:7b-instruct-q4_0` may be added later as a text-only mathematics specialist. It can be compared on the 17 non-image items, but its results must not be placed in the same 20-item overall ranking as the multimodal models.

## 10. Raw Runs and Result Immutability

Every attempt will receive a unique `run_id` and be written before evaluation. The raw record will contain:

- `run_id`, `problem_id`, benchmark version, and prompt version.
- Provider, exact model tag, and model digest.
- Generation parameters and supported seed behavior.
- Full rendered prompt and paths or hashes for image inputs.
- Unmodified model response.
- Start and completion timestamps and elapsed time.
- Attempt number and completion status.
- Technical error type and message, if applicable.

Raw attempt files are append-only. A retry creates a new attempt linked to the original; it never overwrites it. A run manifest records the intended sample, completed attempts, software versions, operating system, CPU, GPU, RAM, and Ollama version.

## 11. Deterministic Final-Answer Validation

The answer parser will first look for the required `Final answer:` marker. If absent, it may propose a conservative extraction from the final non-empty line, but that extraction must be marked for human confirmation.

Validators will be selected by `answer_type`:

- Exact normalized string or choice.
- Integer or rational equality.
- Floating-point comparison with explicit absolute and relative tolerances.
- Symbolic equivalence under documented domain assumptions.
- Equation-solution, finite-set, and interval equivalence.
- Structured-text comparison for narrowly specified logical conclusions.

Normalization may standardize whitespace, Unicode minus signs, harmless delimiters, and common mathematical formatting. It must not silently repair a substantive mathematical error.

Symbolic parsing will use a restricted grammar and an allowlist of mathematical names. Model output will never be passed to Python `eval`. Symbolic work will be bounded by input length and execution time. Ambiguous or unsupported cases go to manual review rather than being forced into a Boolean decision.

`final_answer_correct` is normally produced by the validator. A human override is allowed only with an explicit reason, original automated result, reviewer identifier, and timestamp.

## 12. Human Evaluation Rubric

### Correctness: 0–4

- **4 — Fully correct:** all material claims, calculations, and the conclusion are correct.
- **3 — Mostly correct:** the method is valid and the response has only a localized minor error or omission.
- **2 — Partially correct:** meaningful progress is present, but a major error prevents a correct solution.
- **1 — Minimally correct:** the response engages with the task but the approach is largely invalid or undeveloped.
- **0 — Incorrect or absent:** no meaningful correct progress, irrelevant content, or no solution.

### Reasoning Quality: 0–4

- **4 — Excellent:** coherent, sufficient, well-justified reasoning with no material gap.
- **3 — Sound:** valid reasoning with a minor clarity issue or nonessential omitted step.
- **2 — Mixed:** some valid reasoning, but an important gap, unsupported transition, or inconsistent step.
- **1 — Poor:** major logical defects or mostly unsupported assertions.
- **0 — None:** no usable reasoning or wholly incoherent reasoning.

### Instruction Following: 0–2

- **2 — Complete:** follows the requested format and all task-specific constraints.
- **1 — Partial:** the mathematical task is attempted but one noncritical instruction is missed.
- **0 — Failed:** ignores a material constraint, omits the requested answer format without a recoverable conclusion, or refuses without cause.

The derived `fully_correct_solution` flag is true only when the final answer is correct, correctness is 4, reasoning quality is at least 3, and instruction following is 2. This definition will be fixed before the primary run.

## 13. Error Taxonomy

Each response will receive exactly one primary label and may receive one secondary label:

- **Correct:** no material mathematical or instructional error.
- **Arithmetic or algebraic error:** an invalid calculation or symbolic manipulation causes the failure.
- **Logical inference error:** the conclusion does not follow from stated premises or prior steps.
- **Incorrect assumption:** the response introduces an unjustified domain, independence, regularity, or modeling assumption.
- **Problem misinterpretation:** the response solves a materially different task or misreads a graph, condition, quantity, or requested output.
- **Incomplete reasoning:** the approach may be promising, but required steps or the conclusion are missing.
- **Unsupported claim or hallucination:** the response invents a theorem, datum, graph feature, condition, or external fact.
- **Instruction-following failure:** the mathematics is otherwise usable, but a material response constraint is violated.

The primary error is the earliest causal error that materially changes or prevents the solution. Later consequences of that error are not counted as separate primaries. `first_error_step` will quote or precisely locate the relevant step, while review notes explain the classification.

## 14. Annotation Quality Control

The annotation guide will include a decision tree, rubric anchors, and representative examples. Evaluation will be performed without viewing aggregate model rankings.

A stratified 20% sample of responses will be evaluated a second time in a delayed, blinded pass. The report will provide exact and adjacent-score intra-rater agreement and list adjudicated changes. Cohen's kappa will be reported only if a genuinely independent second reviewer participates; it will not be simulated for a single reviewer.

Low-confidence annotations must include a note and receive a second review. Changes to evaluations remain auditable through timestamps or version-control history.

## 15. Metrics and Statistical Interpretation

The primary report will include:

- Final-answer accuracy.
- Fully correct solution rate.
- Mean and distribution of correctness, reasoning-quality, and instruction-following scores.
- Accuracy by category and difficulty.
- Primary error-category counts and proportions.
- Answer-extraction failure rate.
- Technical failure rate.
- Latency as an operational, non-quality metric.

No single composite score will be used. It would conceal meaningful differences between arriving at the correct result, justifying it, and following instructions.

Overall binary proportions will include 95% Wilson intervals. Score means may include bootstrap intervals, with the resampling method and seed documented. Category-level results contain only three or four items and will be labeled descriptive and exploratory. The report will not use significance language or claim universal superiority from this sample.

Planned figures are:

1. Overall accuracy with confidence intervals.
2. Model-by-category accuracy heatmap with item counts shown.
3. Performance by difficulty.
4. Stacked distribution of primary error categories.
5. Rubric-score distributions by model.

## 16. Technical Failures and Recovery

Attempt statuses will distinguish `success`, `provider_error`, `timeout`, `invalid_response`, and `cancelled`.

- A timeout or unreachable Ollama server is a technical failure and is excluded from mathematical-accuracy denominators while being reported separately.
- A successfully returned response with no usable answer is a model failure, not an infrastructure failure. It remains in the denominator and is evaluated for instruction following.
- One automatic retry is allowed for transient provider errors and timeouts. Both attempts remain preserved.
- Completed responses are skipped on resume unless the user explicitly requests a new run.
- Dataset or prompt version mismatches stop the run before inference.
- Analysis refuses to combine incompatible benchmark versions in one primary comparison.

## 17. Analysis and Reporting Outputs

Pandas will be the canonical analysis layer. Analysis functions will return tables before plotting so they can be unit-tested. Charts will be produced from those tables with consistent labels, accessible colors, visible sample sizes, and no truncated axes that distort comparisons.

The final report will contain:

1. Research question and benchmark scope.
2. Problem construction and verification methods.
3. Model and inference configuration.
4. Evaluation rubric and error taxonomy.
5. Primary results.
6. Qualitative case studies showing representative successes and failures.
7. Stability-probe findings.
8. Annotation quality-control results.
9. Limitations and threats to validity.
10. Reproduction instructions and possible extensions.

The repository will include a report template before inference. It will not include fabricated model findings. Test fixtures and demonstration charts must be visibly labeled synthetic and confined to test or example paths.

## 18. Testing and Continuous Integration

The supported Python version will be 3.11 or newer. Core dependencies will be kept small and include Pandas, Pydantic, SymPy, a plotting library, YAML support, and a command-line library. Provider-specific packages will be optional dependencies where practical.

The test suite will cover:

- Schema validity, unique IDs, and the exact benchmark distribution.
- Figure and source-data path integrity.
- Ground-truth acceptance by every configured validator.
- Rejection of curated near-miss answers for each validator class.
- Answer extraction and normalization edge cases.
- Restricted symbolic parsing and resource limits.
- Provider-contract behavior through mocks.
- Retry, resume, and immutable raw-attempt behavior.
- Rubric-field and error-taxonomy constraints.
- Metric calculations against small hand-computed fixtures.
- Command-line smoke tests.

Tests that need Ollama will be marked as integration tests and skipped by default. GitHub Actions will run the complete non-Ollama suite on every push. CI will never download models or require API credentials.

## 19. Portability and Target-PC Execution

The current development machine does not need Ollama or model weights. The target Windows PC will install the latest supported Ollama release, then download:

```text
ollama pull qwen2.5vl:3b
ollama pull gemma3:4b
```

The target PC should have an up-to-date NVIDIA driver, sufficient free disk space for Ollama and at least 6.5 GB of model files, and Python 3.11 or newer. The setup guide will include verification commands, environment capture, a one-item smoke run, the full run, evaluation, and report generation.

No secrets are required for local execution. Optional future APIs will read credentials from environment variables and `.env` files excluded by version control. Raw prompts and responses will contain no private user data.

## 20. Delivery Milestones

### Milestone A — Implementation Without Local Models

- Repository structure, package configuration, and documentation exist.
- The 20-item English benchmark and all reference solutions are complete.
- Three original graph images and their source data are committed.
- Dataset validation and all deterministic validators pass.
- The mock provider, Ollama provider, CLI, evaluation schema, and analysis pipeline are implemented.
- Unit tests and non-Ollama CI pass.
- The annotation guide and empty report template are complete.

### Milestone B — Target-PC Experiment

- Ollama and both exact model artifacts are installed on the target PC.
- Smoke tests succeed for text and image input.
- The 40-response primary run completes with a saved manifest.
- Human evaluation and the blinded quality-control sample are complete.
- The stability probe is completed and kept separate from primary results.
- All tables, charts, and the final report are regenerated from real saved evaluations.
- The README states the exact reproduction commands and measured limitations.

The portfolio is implementation-complete after Milestone A but not empirically complete until Milestone B. No benchmark conclusions will be published before Milestone B.

## 21. Limitations

The report must state at least these limitations:

- Twenty items provide broad coverage but low statistical power.
- Category subsets of three or four items cannot support strong rankings.
- Manual scoring can contain evaluator bias despite anchored rubrics and a blinded second pass.
- Quantized small models do not represent all local or commercial LLMs.
- Temperature-zero generation may still vary across software or hardware versions.
- Model contamination cannot be ruled out completely, even for newly authored items.
- Graph tasks combine visual perception and mathematical reasoning, so failures may not isolate one capability.
- Latency measurements are hardware-specific.

## 22. Deferred Extensions

The following are intentionally deferred until the 20-item benchmark is complete:

- Expansion to 100 problems.
- Paid OpenAI, Anthropic, or Gemini provider adapters.
- Additional local models or model-size scaling experiments.
- Independent multi-annotator agreement studies.
- A hosted dashboard or web interface.
- Automated LLM-as-judge experiments, which would remain secondary to human evaluation.

## 23. Approved Design Decisions

- Portfolio target: Research Analyst – Advanced Math.
- Deliverables: entirely in English.
- Initial scope: 20 problems, not 100.
- Execution: free and local-first, with optional APIs later.
- Evaluation: deterministic plus structured human review.
- Primary models: `qwen2.5vl:3b` and `gemma3:4b` through Ollama.
- Development machine: no Ollama or model installation required.
- Target hardware: 16 GB RAM and an NVIDIA RTX 2060 Super-class GPU.
- Results: real inference only; fixtures are never presented as experimental findings.
