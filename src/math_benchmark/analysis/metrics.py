import math

import pandas as pd  # type: ignore[import-untyped]


class AnalysisError(ValueError):
    """Analysis inputs are incomplete or combine incompatible experiments."""


def _ensure_single_experiment(frame: pd.DataFrame) -> None:
    if "experiment_kind" not in frame:
        return
    kinds = set(frame["experiment_kind"].dropna().astype(str))
    if len(kinds) > 1:
        raise AnalysisError(f"cannot combine experiment kinds by default: {sorted(kinds)}")


def _truthy(series: pd.Series) -> pd.Series:
    return series.map(
        lambda value: value is True
        or (isinstance(value, (int, float)) and value == 1)
        or str(value).strip().lower() in {"true", "1"}
    )


def _scored(frame: pd.DataFrame) -> pd.DataFrame:
    scored = frame.copy()
    if "status" in scored:
        scored = scored[scored["status"].astype(str) == "success"]
    return scored[scored["correctness_score"].notna()]


def wilson_interval(
    successes: int, total: int, z: float = 1.959963984540054
) -> tuple[float, float]:
    if total <= 0 or successes < 0 or successes > total:
        raise AnalysisError("Wilson interval requires 0 <= successes <= total and total > 0")
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2))
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def overall_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    _ensure_single_experiment(frame)
    rows: list[dict[str, object]] = []
    for model_name, all_model_rows in frame.groupby("model_name", sort=True):
        scored = _scored(all_model_rows)
        n_scored = len(scored)
        technical = len(all_model_rows) - n_scored
        correct = int(_truthy(scored["final_answer_correct"]).sum())
        fully_correct = int(_truthy(scored["fully_correct_solution"]).sum())
        lower, upper = wilson_interval(correct, n_scored) if n_scored else (math.nan, math.nan)
        rows.append(
            {
                "model_name": model_name,
                "n": n_scored,
                "n_scored": n_scored,
                "n_technical_failures": technical,
                "final_answer_accuracy": correct / n_scored if n_scored else math.nan,
                "final_answer_accuracy_low": lower,
                "final_answer_accuracy_high": upper,
                "fully_correct_rate": fully_correct / n_scored if n_scored else math.nan,
                "mean_correctness": pd.to_numeric(scored["correctness_score"]).mean(),
                "mean_reasoning_quality": pd.to_numeric(
                    scored["reasoning_quality_score"]
                ).mean(),
                "mean_instruction_following": pd.to_numeric(
                    scored["instruction_following_score"]
                ).mean(),
            }
        )
    return pd.DataFrame(rows)


def _grouped_accuracy(frame: pd.DataFrame, group_column: str) -> pd.DataFrame:
    _ensure_single_experiment(frame)
    rows: list[dict[str, object]] = []
    for (model_name, group_value), group in _scored(frame).groupby(
        ["model_name", group_column], sort=True
    ):
        n = len(group)
        correct = int(_truthy(group["final_answer_correct"]).sum())
        lower, upper = wilson_interval(correct, n)
        rows.append(
            {
                "model_name": model_name,
                group_column: group_value,
                "n": n,
                "final_answer_accuracy": correct / n,
                "accuracy_low": lower,
                "accuracy_high": upper,
            }
        )
    return pd.DataFrame(rows)


def metrics_by_category(frame: pd.DataFrame) -> pd.DataFrame:
    result = _grouped_accuracy(frame, "category")
    result["category_label"] = result.apply(
        lambda row: f"{row['category']} (n={int(row['n'])})", axis=1
    )
    return result


def metrics_by_difficulty(frame: pd.DataFrame) -> pd.DataFrame:
    return _grouped_accuracy(frame, "difficulty")


def error_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    _ensure_single_experiment(frame)
    scored = _scored(frame)
    counts = (
        scored.groupby(["model_name", "primary_error"], dropna=False, sort=True)
        .size()
        .rename("n")
        .reset_index()
    )
    totals = counts.groupby("model_name")["n"].transform("sum")
    counts["proportion"] = counts["n"] / totals
    return counts


def _agreement(first: pd.Series, second: pd.Series) -> float:
    return float((first.to_numpy() == second.to_numpy()).mean())


def _within_one(first: pd.Series, second: pd.Series) -> float:
    left = pd.to_numeric(first).to_numpy()
    right = pd.to_numeric(second).to_numpy()
    return float((abs(left - right) <= 1).mean())


def _cohen_kappa(first: pd.Series, second: pd.Series) -> float:
    labels = sorted(set(first.astype(str)) | set(second.astype(str)))
    observed = _agreement(first.astype(str), second.astype(str))
    expected = sum(
        float((first.astype(str) == label).mean())
        * float((second.astype(str) == label).mean())
        for label in labels
    )
    return 1.0 if expected == 1.0 and observed == 1.0 else (observed - expected) / (1 - expected)


def annotation_agreement(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"blinded_review_id", "pass"}
    if not required.issubset(frame.columns):
        raise AnalysisError(f"agreement data missing columns: {sorted(required - set(frame.columns))}")
    original = frame[frame["pass"] == "original"].set_index("blinded_review_id")
    recheck = frame[frame["pass"] == "recheck"].set_index("blinded_review_id")
    shared = original.index.intersection(recheck.index)
    if shared.empty:
        raise AnalysisError("agreement data has no paired reviews")
    original, recheck = original.loc[shared], recheck.loc[shared]
    row: dict[str, object] = {
        "n_pairs": len(shared),
        "agreement_type": "intra-rater"
        if len(set(frame["reviewer_id"].dropna().astype(str))) <= 1
        else "inter-rater",
        "correctness_exact_agreement": _agreement(
            original["correctness_score"], recheck["correctness_score"]
        ),
        "correctness_within_one_agreement": _within_one(
            original["correctness_score"], recheck["correctness_score"]
        ),
        "reasoning_exact_agreement": _agreement(
            original["reasoning_quality_score"], recheck["reasoning_quality_score"]
        ),
        "reasoning_within_one_agreement": _within_one(
            original["reasoning_quality_score"], recheck["reasoning_quality_score"]
        ),
        "instruction_exact_agreement": _agreement(
            original["instruction_following_score"], recheck["instruction_following_score"]
        ),
        "primary_error_agreement": _agreement(
            original["primary_error"], recheck["primary_error"]
        ),
    }
    if row["agreement_type"] == "inter-rater":
        row["cohen_kappa"] = _cohen_kappa(
            original["primary_error"], recheck["primary_error"]
        )
    return pd.DataFrame([row])

