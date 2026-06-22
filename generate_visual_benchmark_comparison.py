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
import numpy as np
import zarr


DATASETS = [
    (
        "canola",
        "Canola",
        "canola.ome.zarr",
        "canola_variants_benchmark.csv",
        "canola_variants_report.md",
    ),
    (
        "beef",
        "Beef",
        "Beef_fixed.ome.zarr",
        "beef_variants_benchmark.csv",
        "beef_variants_report.md",
    ),
    (
        "geo",
        "Geo",
        "Geo.ome.zarr",
        "geo_variants_benchmark.csv",
        "geo_variants_report.md",
    ),
]
FORMATS = ["matlab_compressed", "envi", "npy", "omezarr"]
FORMAT_LABELS = {
    "matlab_compressed": "MATLAB\ncompressed",
    "envi": "ENVI",
    "npy": "NumPy",
    "omezarr": "OME-Zarr",
}
FORMAT_COLORS = {
    "matlab_compressed": "#b85c38",
    "envi": "#7d6a0d",
    "npy": "#4a7c59",
    "omezarr": "#1f6f8b",
}
FORMAT_KEYS = {
    "MATLAB compressed": "matlab_compressed",
    "ENVI": "envi",
    "NumPy": "npy",
    "OME-Zarr": "omezarr",
}
SIZE_PATTERN = re.compile(r"^\| (?P<label>[^|]+) \| (?P<size>[0-9.]+) MiB \|")
BACKGROUND = "#f4f0e8"
PANEL = "#fbfaf7"
GRID_COLOR = "#d7d7d0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a visual benchmark comparison figure for the paper."
    )
    parser.add_argument(
        "--base-dir",
        default=Path(__file__).parent,
        type=Path,
        help="Directory containing benchmark outputs and OME-Zarr datasets.",
    )
    parser.add_argument(
        "--out",
        default="visual_benchmark_comparison.png",
        help="Output PNG filename.",
    )
    return parser.parse_args()


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": BACKGROUND,
            "savefig.facecolor": BACKGROUND,
            "axes.facecolor": PANEL,
            "axes.edgecolor": "#404040",
            "font.size": 10,
        }
    )


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


def normalize(image: np.ndarray) -> np.ndarray:
    lower = float(np.nanpercentile(image, 2))
    upper = float(np.nanpercentile(image, 98))
    return np.clip((image - lower) / max(upper - lower, 1e-9), 0, 1)


def display_level(root) -> np.ndarray:
    level_names = sorted(
        [name for name in root.array_keys() if name.isdigit()],
        key=lambda item: int(item),
    )
    for name in reversed(level_names):
        arr = root[name]
        if arr.shape[1] <= 900 and arr.shape[2] <= 2400:
            return arr
    return root[level_names[-1]]


def false_color(root) -> np.ndarray:
    arr = display_level(root)
    bands = [
        min(int(round(arr.shape[0] * 0.20)), arr.shape[0] - 1),
        min(int(round(arr.shape[0] * 0.50)), arr.shape[0] - 1),
        min(int(round(arr.shape[0] * 0.80)), arr.shape[0] - 1),
    ]
    channels = [normalize(np.asarray(arr[index, :, :], dtype=np.float32)) for index in bands]
    return np.dstack(channels[::-1])


def dataset_metrics(base_dir: Path, csv_name: str, report_name: str) -> dict[str, dict[str, float]]:
    series = read_csv(base_dir / csv_name)
    sizes = read_sizes(base_dir / report_name)
    metrics: dict[str, dict[str, float]] = {}
    for format_key in FORMATS:
        metrics[format_key] = {
            "size_mib": sizes[format_key],
            "open": average_metric(series, f"{format_key}_open", "open_seconds"),
            "navigation": average_metric(
                series, f"{format_key}_random_navigation", "per_step_seconds"
            ),
            "pca": average_metric(series, f"{format_key}_pca", "total_seconds"),
        }
    return metrics


def plot_metric_row(ax, metrics: dict[str, dict[str, float]], metric: str, ylabel: str) -> None:
    values = [metrics[format_key][metric] for format_key in FORMATS]
    labels = [FORMAT_LABELS[format_key] for format_key in FORMATS]
    colors = [FORMAT_COLORS[format_key] for format_key in FORMATS]
    ax.bar(np.arange(len(FORMATS)), values, color=colors, width=0.72)
    ax.set_xticks(np.arange(len(FORMATS)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_yscale("log")
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, which="both")
    ax.set_axisbelow(True)


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir
    output_path = Path(args.out)
    if not output_path.is_absolute():
        output_path = base_dir / output_path

    setup_style()
    fig = plt.figure(figsize=(13.5, 12.5))
    grid = fig.add_gridspec(
        4,
        3,
        height_ratios=[1.35, 1.0, 1.0, 1.0],
        hspace=0.45,
        wspace=0.24,
    )
    fig.suptitle(
        "Visual Benchmark Comparison Across Hyperspectral Dataset Sizes",
        fontsize=16,
        weight="bold",
    )

    for col, (_, dataset_label, zarr_name, csv_name, report_name) in enumerate(DATASETS):
        root = zarr.open(str(base_dir / zarr_name), mode="r")
        arr0 = root["0"]
        metrics = dataset_metrics(base_dir, csv_name, report_name)

        image_ax = fig.add_subplot(grid[0, col])
        image_ax.imshow(false_color(root), aspect="auto")
        image_ax.set_title(
            f"{dataset_label}\n{arr0.shape[0]} bands, {arr0.shape[1]} x {arr0.shape[2]} pixels",
            fontsize=11,
            weight="bold",
        )
        image_ax.set_xticks([])
        image_ax.set_yticks([])
        for spine in image_ax.spines.values():
            spine.set_visible(False)

        open_ax = fig.add_subplot(grid[1, col])
        plot_metric_row(open_ax, metrics, "open", "Open time (s)")

        nav_ax = fig.add_subplot(grid[2, col])
        plot_metric_row(nav_ax, metrics, "navigation", "Random nav.\n(s/step)")

        pca_ax = fig.add_subplot(grid[3, col])
        plot_metric_row(pca_ax, metrics, "pca", "PCA time (s)")

    fig.text(
        0.5,
        0.025,
        "Top row shows representative false-color dataset views. Bar plots compare compressed MATLAB, ENVI, NumPy, and OME-Zarr; y-axes are log-scaled.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.95))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Wrote {output_path.resolve()}")


if __name__ == "__main__":
    main()
