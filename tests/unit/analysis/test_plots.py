from pathlib import Path

import matplotlib.image as mpimg
import pandas as pd

from math_benchmark.analysis.metrics import (
    error_distribution,
    metrics_by_category,
    metrics_by_difficulty,
    overall_metrics,
)
from math_benchmark.analysis.plots import create_all_plots

FIXTURE = Path(__file__).parents[2] / "fixtures" / "analysis" / "evaluations.csv"


def test_create_exactly_five_nonempty_plots(tmp_path: Path) -> None:
    frame = pd.read_csv(FIXTURE)
    tables = {
        "overall": overall_metrics(frame),
        "category": metrics_by_category(frame),
        "difficulty": metrics_by_difficulty(frame),
        "errors": error_distribution(frame),
    }
    paths = create_all_plots(frame, tables, tmp_path)
    assert [path.name for path in paths] == [
        "overall_accuracy.png",
        "category_accuracy_heatmap.png",
        "difficulty_accuracy.png",
        "error_distribution.png",
        "rubric_scores.png",
    ]
    assert all(path.stat().st_size > 0 for path in paths)
    assert all(all(size > 0 for size in mpimg.imread(path).shape[:2]) for path in paths)
    assert all("(n=" in label for label in tables["category"]["category_label"])

