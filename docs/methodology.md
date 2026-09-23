# Methodology

## Study design and freeze

Benchmark version 1.0.0 freezes 20 English-language items before primary inference. The exact category distribution is algebra 4, calculus 4, probability 3, logic 3, graph interpretation 3, and word problems 3. Difficulty totals are foundational 5, intermediate 9, and advanced 6. Wording, metadata, ground-truth, or figure changes require a version increment. Pilot outputs are excluded from comparison.

## Models and prompts

The primary comparison uses `qwen2.5vl:3b` and `gemma3:4b` through Ollama, one model at a time. Every problem begins a stateless conversation with prompt version 1.0.0:

```text
Provide a concise solution that explains the key steps.
End your response with: Final answer: <answer>
```

Problem-specific units and formatting remain in the item text. Exact model tags and content digests are captured before inference.

## Generation parameters

The primary profile uses temperature 0, seed 42, context length 4096, maximum prediction length 2048, a 300-second timeout, and one repetition. The stability profile uses temperature 0.2 and seeds 1103, 2207, and 3301 on five fixed items, producing 30 attempts across two models. Primary and stability records are analyzed separately.

## Validation and annotation

Final answers are extracted from a line beginning with `Final answer:`. Missing or conflicting markers are flagged for review. Deterministic validators cover exact text, rational equivalence, explicit numeric tolerances, restricted symbolic equivalence, sets/intervals, and structured JSON. Symbolic work uses an allowlist, 512-character limit, and a bounded child process; model text is never passed to Python `eval`.

Successful provider responses remain in the mathematical denominator even when they lack a usable answer. Timeouts, provider failures, and invalid provider-protocol responses are technical failures and are reported separately. A human may override automation only with the original result, reason, reviewer ID, and timestamp.

Human evaluation assigns correctness 0–4, reasoning quality 0–4, instruction following 0–2, one primary error, and at most one secondary error. The primary error is the earliest causal failure. `fully_correct_solution` requires a correct final answer, correctness 4, reasoning quality at least 3, and instruction following 2.

## Metrics and uncertainty

Reported quantities include final-answer accuracy, fully-correct rate, mean rubric scores, category and difficulty breakdowns, technical-failure counts, and error distributions. Every grouped table carries its item count. Accuracy uses a 95% Wilson score interval with `z=1.959963984540054`; no normal approximation or SciPy dependency is used.

A delayed, blinded, model-by-difficulty stratified 20% recheck reports exact and within-one-point agreement for correctness and reasoning, exact instruction-following agreement, and primary-error agreement. Cohen's kappa is reported only when two distinct reviewers genuinely participate.

## Reproducibility controls

Raw attempts are append-only JSONL with UUID run IDs and parent IDs for retries. Prompts and image bytes are SHA-256 hashed. Manifests record benchmark/prompt versions, model tags/digests, generation parameters, environment, package versions, OS, CPU, RAM, GPU when available, and Ollama version. CI rebuilds the canonical JSONL byte for byte, verifies frozen figure hashes, and checks that figure generation is deterministic within one software environment. The committed PNGs remain the canonical inputs because Matplotlib raster encoding can vary across operating systems.

## Limitations

The benchmark is intentionally small, authored, and optimized for auditability rather than population-level coverage. Category estimates have low sample sizes. Two compact local models do not represent the LLM landscape. Human judgment remains imperfect despite explicit anchors and rechecks. Hardware, quantization, runtime version, and model artifact updates may affect outputs. Graph interpretation includes only three generated figures. No experimental claim should be made from committed synthetic fixtures.
