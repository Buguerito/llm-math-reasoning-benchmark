from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd  # type: ignore[import-untyped]

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
COLORS = [BLUE, ORANGE, GREEN, PURPLE]


def _save(figure: plt.Figure, path: Path) -> None:  # type: ignore[name-defined]
    figure.tight_layout()
    figure.savefig(path, dpi=120, metadata={"Software": "llm-math-reasoning-benchmark"})
    plt.close(figure)


def create_all_plots(
    evaluations: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        output_dir / "overall_accuracy.png",
        output_dir / "category_accuracy_heatmap.png",
        output_dir / "difficulty_accuracy.png",
        output_dir / "error_distribution.png",
        output_dir / "rubric_scores.png",
    ]

    overall = tables["overall"]
    figure, axes = plt.subplots(figsize=(8, 5))
    x = range(len(overall))
    axes.bar([value - 0.18 for value in x], overall["final_answer_accuracy"], 0.36, label="Final answer", color=BLUE)
    axes.bar([value + 0.18 for value in x], overall["fully_correct_rate"], 0.36, label="Fully correct", color=ORANGE)
    axes.set(xticks=list(x), xticklabels=overall["model_name"], ylim=(0, 1), ylabel="Proportion", title="Overall Accuracy")
    axes.legend()
    _save(figure, paths[0])

    category = tables["category"]
    pivot = category.pivot(index="model_name", columns="category_label", values="final_answer_accuracy")
    figure, axes = plt.subplots(figsize=(10, 5))
    image = axes.imshow(pivot.to_numpy(), vmin=0, vmax=1, cmap="Blues", aspect="auto")
    axes.set(yticks=range(len(pivot.index)), yticklabels=pivot.index)
    axes.set(xticks=range(len(pivot.columns)), xticklabels=pivot.columns, title="Accuracy by Category")
    axes.tick_params(axis="x", rotation=30)
    for row in range(len(pivot.index)):
        for column in range(len(pivot.columns)):
            value = pivot.iloc[row, column]
            axes.text(column, row, f"{value:.2f}", ha="center", va="center")
    figure.colorbar(image, ax=axes, label="Accuracy")
    _save(figure, paths[1])

    difficulty = tables["difficulty"]
    difficulties = [value for value in ("foundational", "intermediate", "advanced") if value in set(difficulty["difficulty"])]
    models = sorted(difficulty["model_name"].unique())
    figure, axes = plt.subplots(figsize=(8, 5))
    width = 0.8 / max(1, len(models))
    for model_index, model in enumerate(models):
        subset = difficulty[difficulty["model_name"] == model].set_index("difficulty")
        positions = [index - 0.4 + width / 2 + model_index * width for index in range(len(difficulties))]
        axes.bar(positions, [subset.loc[item, "final_answer_accuracy"] for item in difficulties], width, label=model, color=COLORS[model_index % len(COLORS)])
    axes.set(xticks=range(len(difficulties)), xticklabels=difficulties, ylim=(0, 1), ylabel="Accuracy", title="Accuracy by Difficulty")
    axes.legend()
    _save(figure, paths[2])

    errors = tables["errors"]
    error_pivot = errors.pivot(index="primary_error", columns="model_name", values="proportion").fillna(0)
    figure, axes = plt.subplots(figsize=(10, 5))
    error_pivot.plot(kind="bar", ax=axes, color=COLORS[: len(error_pivot.columns)])
    axes.set(ylim=(0, 1), ylabel="Proportion", xlabel="Primary error", title="Error Distribution")
    axes.tick_params(axis="x", rotation=30)
    _save(figure, paths[3])

    scored = evaluations[evaluations["correctness_score"].notna()].copy()
    rubric = scored.groupby("model_name")[["correctness_score", "reasoning_quality_score", "instruction_following_score"]].mean()
    figure, axes = plt.subplots(figsize=(8, 5))
    rubric.plot(kind="bar", ax=axes, color=COLORS[:3])
    axes.set(ylabel="Mean score", xlabel="Model", title="Rubric Scores", ylim=(0, 4))
    axes.tick_params(axis="x", rotation=0)
    _save(figure, paths[4])
    return paths

