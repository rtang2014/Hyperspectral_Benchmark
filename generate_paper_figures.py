import argparse
import csv
import os
import re
from pathlib import Path


def configure_runtime() -> None:
    cache_root = Path(".cache")
    cache_home_dir = cache_root / "home"
    mpl_cache_dir = cache_root / "matplotlib"
    xdg_cache_dir = cache_root / "xdg"
    cache_home_dir.mkdir(parents=True, exist_ok=True)
    mpl_cache_dir.mkdir(parents=True, exist_ok=True)
    xdg_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HOME"] = str(cache_home_dir.resolve())
    os.environ["MPLCONFIGDIR"] = str(mpl_cache_dir.resolve())
    os.environ["XDG_CACHE_HOME"] = str(xdg_cache_dir.resolve())


configure_runtime()

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


DATASETS = [
    ("canola", "Canola", "canola_variants_benchmark.csv", "canola_variants_report.md"),
    ("beef", "Beef", "beef_variants_benchmark.csv", "beef_variants_report.md"),
    ("geo", "Geo", "geo_variants_benchmark.csv", "geo_variants_report.md"),
]
MAIN_FORMATS = ["matlab_compressed", "envi", "npy", "omezarr"]
ALL_FORMATS = ["matlab_compressed", "matlab_uncompressed", "envi", "npy", "omezarr"]
FORMAT_LABELS = {
    "matlab_compressed": "MATLAB compressed",
    "matlab_uncompressed": "MATLAB uncompressed",
    "envi": "ENVI",
    "npy": "NumPy",
    "omezarr": "OME-Zarr",
}
FORMAT_KEYS = {label: key for key, label in FORMAT_LABELS.items()}
FORMAT_COLORS = {
    "matlab_compressed": "#b85c38",
    "matlab_uncompressed": "#8f3d21",
    "envi": "#7d6a0d",
    "npy": "#4a7c59",
    "omezarr": "#1f6f8b",
}
SCENARIOS = [
    ("open", "open_seconds", "Initial Open Time", "Seconds"),
    (
        "sequential_navigation",
        "per_step_seconds",
        "Sequential Navigation",
        "Seconds per step",
    ),
    ("random_navigation", "per_step_seconds", "Random Navigation", "Seconds per step"),
    ("pca", "total_seconds", "PCA End-to-End Time", "Seconds"),
]
BACKGROUND = "#f4f0e8"
PANEL_BACKGROUND = "#fbfaf7"
GRID_COLOR = "#d7d7d0"
TEXT_COLOR = "#242424"
SIZE_PATTERN = re.compile(r"^\| (?P<label>[^|]+) \| (?P<size>[0-9.]+) MiB \|")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate paper figures from hyperspectral benchmark outputs."
    )
    parser.add_argument(
        "--base-dir",
        default=Path(__file__).parent,
        type=Path,
        help="Directory containing benchmark CSV and report files.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        type=Path,
        help="Directory for generated figures. Defaults to base-dir.",
    )
    return parser.parse_args()


def read_sizes(report_path: Path) -> dict[str, float]:
    sizes: dict[str, float] = {}
    for line in report_path.read_text().splitlines():
        match = SIZE_PATTERN.match(line)
        if not match:
            continue
        key = FORMAT_KEYS.get(match.group("label").strip())
        if key is not None:
            sizes[key] = float(match.group("size"))
    return sizes


def read_csv(csv_path: Path) -> dict[str, list[dict[str, float]]]:
    series: dict[str, list[dict[str, float]]] = {}
    with csv_path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            metrics: dict[str, float] = {}
            for key, value in row.items():
                if key in {"series", "run"} or value in {"", None}:
                    continue
                metrics[key] = float(value)
            series.setdefault(row["series"], []).append(metrics)
    return series


def average_metric(series: dict[str, list[dict[str, float]]], label: str, metric: str) -> float:
    values = [row[metric] for row in series[label] if metric in row]
    return sum(values) / len(values)


def collect_points(base_dir: Path, format_order: list[str]) -> list[dict[str, object]]:
    points: list[dict[str, object]] = []
    for dataset_key, dataset_label, csv_name, report_name in DATASETS:
        sizes = read_sizes(base_dir / report_name)
        series = read_csv(base_dir / csv_name)
        for format_key in format_order:
            for scenario, metric, _, _ in SCENARIOS:
                series_name = f"{format_key}_{scenario}"
                if format_key not in sizes or series_name not in series:
                    continue
                points.append(
                    {
                        "dataset": dataset_key,
                        "dataset_label": dataset_label,
                        "format": format_key,
                        "scenario": scenario,
                        "size_mib": sizes[format_key],
                        "seconds": average_metric(series, series_name, metric),
                    }
                )
    return points


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": BACKGROUND,
            "axes.facecolor": PANEL_BACKGROUND,
            "axes.edgecolor": "#404040",
            "font.size": 10,
            "savefig.facecolor": BACKGROUND,
        }
    )


def save_figure(fig, output_path: Path) -> None:
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path.resolve()}")


def draw_box(ax, xy: tuple[float, float], text: str, width: float = 1.9) -> None:
    x, y = xy
    box = patches.FancyBboxPatch(
        (x - width / 2, y - 0.36),
        width,
        0.72,
        boxstyle="round,pad=0.04,rounding_size=0.04",
        linewidth=1.2,
        edgecolor="#404040",
        facecolor=PANEL_BACKGROUND,
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", color=TEXT_COLOR, fontsize=10)


def draw_arrow(ax, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "->",
            "color": "#404040",
            "linewidth": 1.3,
            "shrinkA": 8,
            "shrinkB": 8,
        },
    )


def plot_workflow(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.6)
    ax.axis("off")
    fig.suptitle("Hyperspectral Format Benchmark Workflow in Python and napari", fontsize=15, weight="bold")

    main_nodes = [
        (1.2, 3.3, "Hyperspectral\ncubes"),
        (3.6, 3.3, "Storage\nformats"),
        (6.0, 3.3, "Standardized\nreader API"),
        (8.4, 3.3, "napari-style\nbenchmark tasks"),
        (10.8, 3.3, "Timing and\nscaling results"),
    ]
    for x, y, text in main_nodes:
        draw_box(ax, (x, y), text)
    for start_x, end_x in [(2.2, 2.6), (4.6, 5.0), (7.0, 7.4), (9.4, 9.8)]:
        draw_arrow(ax, (start_x, 3.3), (end_x, 3.3))

    format_labels = ["MATLAB compressed", "ENVI", "NumPy", "OME-Zarr"]
    for index, label in enumerate(format_labels):
        y = 2.25 - index * 0.42
        ax.text(3.6, y, label, ha="center", va="center", fontsize=9, color=TEXT_COLOR)

    task_labels = [
        "Initial open",
        "Sequential wavelength navigation",
        "Random wavelength navigation",
        "End-to-end PCA",
    ]
    for index, label in enumerate(task_labels):
        y = 2.25 - index * 0.42
        ax.text(8.4, y, label, ha="center", va="center", fontsize=9, color=TEXT_COLOR)

    ax.text(
        6.0,
        0.45,
        "napari was used as the interactive viewing context. All cubes were oriented as (bands, y, x) before timing.",
        ha="center",
        va="center",
        fontsize=9,
        color="#4a4a4a",
    )
    save_figure(fig, out_dir / "figure_1_workflow.png")


def plot_file_sizes(base_dir: Path, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    x = np.arange(len(DATASETS))
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(MAIN_FORMATS))

    for offset, format_key in zip(offsets, MAIN_FORMATS):
        values = []
        for _, _, _, report_name in DATASETS:
            values.append(read_sizes(base_dir / report_name)[format_key])
        ax.bar(
            x + offset,
            values,
            width,
            label=FORMAT_LABELS[format_key],
            color=FORMAT_COLORS[format_key],
        )

    ax.set_title("Stored File Size by Dataset and Format", fontsize=14, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([dataset[1] for dataset in DATASETS])
    ax.set_ylabel("Stored file size (MiB)")
    ax.set_yscale("log")
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, which="both")
    ax.set_axisbelow(True)
    ax.legend(ncol=2, frameon=False)
    save_figure(fig, out_dir / "figure_2_file_size_comparison.png")


def plot_size_trend(base_dir: Path, out_dir: Path, format_order: list[str], filename: str) -> None:
    points = collect_points(base_dir, format_order)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Benchmark Time vs. Stored Image Size", fontsize=16, weight="bold")

    for ax, (scenario, _, title, ylabel) in zip(axes.flat, SCENARIOS):
        scenario_points = [point for point in points if point["scenario"] == scenario]
        for format_key in format_order:
            format_points = [
                point for point in scenario_points if point["format"] == format_key
            ]
            format_points.sort(key=lambda point: point["size_mib"])
            if not format_points:
                continue
            ax.plot(
                [point["size_mib"] for point in format_points],
                [point["seconds"] for point in format_points],
                marker="o",
                linewidth=1.9,
                markersize=6,
                color=FORMAT_COLORS[format_key],
                label=FORMAT_LABELS[format_key],
            )
            for point in format_points:
                ax.annotate(
                    point["dataset_label"],
                    (point["size_mib"], point["seconds"]),
                    xytext=(5, 3),
                    textcoords="offset points",
                    fontsize=8,
                )
        ax.set_title(title, fontsize=13, weight="bold")
        ax.set_xlabel("Stored file size (MiB)")
        ax.set_ylabel(ylabel)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, color=GRID_COLOR, linewidth=0.8, which="both")
        ax.set_axisbelow(True)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(format_order), frameon=False)
    fig.text(
        0.5,
        0.055,
        "Both axes use log scale. Each line connects Canola, Beef, and Geo points for the same storage format.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    save_figure(fig, out_dir / filename)


def plot_task_bars(base_dir: Path, out_dir: Path) -> None:
    points = collect_points(base_dir, MAIN_FORMATS)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.8))
    fig.suptitle("Task-Specific Runtime by Dataset and Format", fontsize=16, weight="bold")
    dataset_labels = [dataset[1] for dataset in DATASETS]
    group_x = np.arange(len(DATASETS))
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(MAIN_FORMATS))

    for ax, (scenario, _, title, ylabel) in zip(axes.flat, SCENARIOS):
        for offset, format_key in zip(offsets, MAIN_FORMATS):
            values = []
            for dataset_key, _, _, _ in DATASETS:
                match = next(
                    point
                    for point in points
                    if point["dataset"] == dataset_key
                    and point["format"] == format_key
                    and point["scenario"] == scenario
                )
                values.append(match["seconds"])
            ax.bar(
                group_x + offset,
                values,
                width,
                color=FORMAT_COLORS[format_key],
                label=FORMAT_LABELS[format_key],
            )
        ax.set_title(title, fontsize=13, weight="bold")
        ax.set_xticks(group_x)
        ax.set_xticklabels(dataset_labels)
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")
        ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, which="both")
        ax.set_axisbelow(True)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    save_figure(fig, out_dir / "figure_4_task_runtime_comparison.png")


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir
    out_dir = args.out_dir or base_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_style()

    plot_workflow(out_dir)
    plot_file_sizes(base_dir, out_dir)
    plot_size_trend(base_dir, out_dir, MAIN_FORMATS, "figure_3_time_vs_size_trend.png")
    plot_task_bars(base_dir, out_dir)
    plot_size_trend(
        base_dir,
        out_dir,
        ALL_FORMATS,
        "supplementary_figure_all_formats_time_vs_size.png",
    )


if __name__ == "__main__":
    main()
