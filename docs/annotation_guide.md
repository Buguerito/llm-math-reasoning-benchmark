# Human Annotation Guide

Annotate responses without viewing aggregate model rankings. Preserve the original response, judge the earliest causal error, and never repair a model answer silently.

## Workflow

1. Read the problem, reference solution, and response.
2. Confirm or override the automated final-answer result.
3. Assign all three rubric scores.
4. Assign exactly one primary error and at most one secondary error.
5. Record the earliest step that materially changes or prevents the solution.
6. Record confidence, notes, reviewer ID, and timestamp.

A human override requires the original automated result, an explicit reason, reviewer ID, and timestamp. Low-confidence annotations require a note and second review.

## Rubric anchors

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

`fully_correct_solution` requires a correct final answer, correctness 4, reasoning quality at least 3, and instruction following 2.

## Error taxonomy and examples

| Label | Correct use | Incorrect use |
|---|---|---|
| `correct` | Every material step and constraint is satisfied. | The numeric answer is right only because two errors cancel. |
| `arithmetic_or_algebraic_error` | The first failure is `3 × 4 = 11`. | A valid calculation follows a false independence assumption. |
| `logical_inference_error` | The response affirms the consequent. | A graph axis is read with the wrong unit. |
| `incorrect_assumption` | Independence is introduced without support. | The given independence is used correctly but multiplication is wrong. |
| `problem_misinterpretation` | The response reports total distance instead of average speed. | It solves the right quantity but makes an arithmetic slip. |
| `incomplete_reasoning` | A promising derivation stops before the requested conclusion. | A conclusion is present but rests on an invented theorem. |
| `unsupported_claim_or_hallucination` | A nonexistent graph point or theorem is asserted. | A known theorem is applied with a small algebra error. |
| `instruction_following_failure` | Correct mathematics is returned in prose when strict JSON was required. | Required JSON is present but contains the wrong value. |

The primary label is the earliest causal error. Later consequences are not new primary errors. Quote or precisely locate that step in `first_error_step`; use notes to explain borderline decisions.

## Confidence

- **3 — High:** the answer and label are clear under the guide.
- **2 — Medium:** a reasonable reviewer could debate a localized judgment.
- **1 — Low:** ambiguity, parsing, or domain interpretation could change the rating; add a note and obtain a second review.

## Blinded recheck

After the first pass, wait before re-annotation and export a deterministic, model-by-difficulty stratified 20% sample. The blinded sheet excludes prior scores, error labels, notes, automated results, and aggregate metrics. Do not reveal rankings during either pass. Compare exact and adjacent-score agreement, adjudicate changes, and report Cohen's kappa only when a genuinely independent second reviewer participated.
