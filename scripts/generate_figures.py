"""Generate deterministic benchmark figures from the committed CSV sources."""

import csv
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

ROOT = Path(__file__).parents[1]
FIGURE_IDS = ("GRF-F-001", "GRF-I-002", "GRF-A-003")
BLUE = "#0072B2"
ORANGE = "#D55E00"
METADATA = {"Software": "llm-math-reasoning-benchmark"}


def _read_rows(path: Path) -> list[list[str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.reader(handle))[1:]


def _configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.3,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _bar_chart(rows: list[list[str]]) -> tuple[Figure, Axes]:
    figure, axes = plt.subplots(figsize=(8, 5))
    axes.bar([row[0] for row in rows], [int(row[1]) for row in rows], color=BLUE)
    axes.set(title="Monthly Sales", xlabel="Month", ylabel="Sales (units)", ylim=(0, 24))
    return figure, axes


def _distance_chart(rows: list[list[str]]) -> tuple[Figure, Axes]:
    figure, axes = plt.subplots(figsize=(8, 5))
    axes.plot(
        [int(row[0]) for row in rows],
        [int(row[1]) for row in rows],
        color=BLUE,
        marker="o",
        linewidth=2.5,
    )
    axes.set(title="Distance Traveled Over Time", xlabel="Time (hours)", ylabel="Distance (km)")
    axes.set(xlim=(-0.2, 6.2), ylim=(-0.5, 13))
    return figure, axes


def _derivative_chart(rows: list[list[str]]) -> tuple[Figure, Axes]:
    figure, axes = plt.subplots(figsize=(8, 5))
    axes.axhline(0, color="#333333", linewidth=1)
    axes.plot(
        [int(row[0]) for row in rows],
        [int(row[1]) for row in rows],
        color=ORANGE,
        marker="o",
        linewidth=2.5,
    )
    axes.set(title="Graph of the Derivative", xlabel="x", ylabel="f'(x)")
    axes.set(xlim=(-0.2, 6.2), ylim=(-2.5, 2.5))
    return figure, axes


PLOTTERS: dict[str, Callable[[list[list[str]]], tuple[Figure, Axes]]] = {
    "GRF-F-001": _bar_chart,
    "GRF-I-002": _distance_chart,
    "GRF-A-003": _derivative_chart,
}


def generate_all(source_dir: Path, output_dir: Path) -> list[Path]:
    """Generate all benchmark figures and return their paths in stable order."""
    _configure()
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    for figure_id in FIGURE_IDS:
        rows = _read_rows(source_dir / f"{figure_id}.csv")
        figure, _ = PLOTTERS[figure_id](rows)
        figure.tight_layout()
        output_path = output_dir / f"{figure_id}.png"
        figure.savefig(output_path, dpi=120, metadata=METADATA)
        plt.close(figure)
        generated.append(output_path)
    return generated


def main() -> None:
    paths = generate_all(ROOT / "data" / "figure_source_data", ROOT / "data" / "figures")
    for path in paths:
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
