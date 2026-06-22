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

import matplotlib.pyplot as plt


DATASETS = [
    ("canola", "Canola", "canola_variants_benchmark.csv", "canola_variants_report.md"),
    ("beef", "Beef", "beef_variants_benchmark.csv", "beef_variants_report.md"),
    ("geo", "Geo", "geo_variants_benchmark.csv", "geo_variants_report.md"),
]
FORMAT_ORDER = [
    "matlab_compressed",
    "envi",
    "npy",
    "omezarr",
]
FORMAT_LABELS = {
    "matlab_compressed": "MATLAB compressed",
    "envi": "ENVI",
    "npy": "NumPy",
    "omezarr": "OME-Zarr",
}
FORMAT_KEYS = {label: key for key, label in FORMAT_LABELS.items()}
FORMAT_COLORS = {
    "matlab_compressed": "#b85c38",
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
GRID_COLOR = "#d7d7d0"
SIZE_PATTERN = re.compile(r"^\| (?P<label>[^|]+) \| (?P<size>[0-9.]+) MiB \|")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot benchmark time versus stored file size across all image sizes."
    )
    parser.add_argument(
        "--base-dir",
        default=Path(__file__).parent,
        type=Path,
        help="Directory containing benchmark CSV and report files.",
    )
    parser.add_argument(
        "--out",
        default="size_time_trend.png",
        help="Output chart image path.",
    )
    return parser.parse_args()


def read_sizes(report_path: Path) -> dict[str, float]:
    sizes: dict[str, float] = {}
    for line in report_path.read_text().splitlines():
        match = SIZE_PATTERN.match(line)
        if not match:
            continue
        label = match.group("label").strip()
        key = FORMAT_KEYS.get(label)
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


def collect_points(base_dir: Path) -> list[dict[str, object]]:
    points: list[dict[str, object]] = []
    for dataset_key, dataset_label, csv_name, report_name in DATASETS:
        sizes = read_sizes(base_dir / report_name)
        series = read_csv(base_dir / csv_name)
        for format_key in FORMAT_ORDER:
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


def plot_scenario(ax, points: list[dict[str, object]], scenario: str, title: str, ylabel: str) -> None:
    scenario_points = [point for point in points if point["scenario"] == scenario]
    for format_key in FORMAT_ORDER:
        format_points = [
            point for point in scenario_points if point["format"] == format_key
        ]
        format_points.sort(key=lambda point: point["size_mib"])
        if not format_points:
            continue
        sizes = [point["size_mib"] for point in format_points]
        seconds = [point["seconds"] for point in format_points]
        ax.plot(
            sizes,
            seconds,
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
    ax.set_facecolor(BACKGROUND)
    ax.grid(True, color=GRID_COLOR, linewidth=0.8, which="both")
    ax.set_axisbelow(True)


def main() -> None:
    args = parse_args()
    points = collect_points(args.base_dir)
    if not points:
        raise ValueError(f"No benchmark points found in {args.base_dir}")

    plt.rcParams.update(
        {
            "figure.facecolor": BACKGROUND,
            "axes.edgecolor": "#404040",
            "font.size": 10,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Benchmark Time vs. Stored Image Size", fontsize=16, weight="bold")
    for ax, (scenario, _, title, ylabel) in zip(axes.flat, SCENARIOS):
        plot_scenario(ax, points, scenario, title, ylabel)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.text(
        0.5,
        0.055,
        "Both axes use log scale. Each line connects Canola, Beef, and Geo points for the same storage format.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))

    output_path = Path(args.out)
    if not output_path.is_absolute():
        output_path = args.base_dir / output_path
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Chart written to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
