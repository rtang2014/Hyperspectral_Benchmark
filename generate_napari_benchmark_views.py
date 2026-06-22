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
from sklearn.decomposition import PCA


DATASETS = [
    (
        "canola",
        "Canola",
        "canola.ome.zarr",
        "canola_variants_benchmark.csv",
        "canola_variants_report.md",
        100,
        128,
    ),
    (
        "beef",
        "Beef",
        "Beef_fixed.ome.zarr",
        "beef_variants_benchmark.csv",
        "beef_variants_report.md",
        240,
        128,
    ),
    (
        "geo",
        "Geo",
        "Geo.ome.zarr",
        "geo_variants_benchmark.csv",
        "geo_variants_report.md",
        80,
        64,
    ),
]
FORMATS = ["matlab_compressed", "envi", "npy", "omezarr"]
FORMAT_LABELS = {
    "matlab_compressed": "MATLAB Compressed",
    "envi": "ENVI",
    "npy": "NumPy",
    "omezarr": "OME-Zarr",
}
FORMAT_KEYS = {label: key for key, label in FORMAT_LABELS.items()}
FORMAT_KEYS["MATLAB compressed"] = "matlab_compressed"
COLORS = {
    "matlab_compressed": "#b85c38",
    "envi": "#9a7d17",
    "npy": "#4a7c59",
    "omezarr": "#1f6f8b",
}
BACKGROUND = "#f4f0e8"
PANEL = "#111111"
TEXT = "#f1efe8"
MUTED = "#d7d0c0"
SIZE_PATTERN = re.compile(r"^\| (?P<label>[^|]+) \| (?P<size>[0-9.]+) MiB \|")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate napari-style benchmark comparison views."
    )
    parser.add_argument(
        "--base-dir",
        default=Path(__file__).parent,
        type=Path,
        help="Directory containing benchmark outputs and OME-Zarr datasets.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        type=Path,
        help="Directory for generated figures. Defaults to base-dir.",
    )
    parser.add_argument(
        "--dataset",
        choices=[dataset[0] for dataset in DATASETS],
        default=None,
        help="Generate only one dataset. Defaults to all datasets.",
    )
    parser.add_argument(
        "--hide-metrics",
        action="store_true",
        help="Hide numeric benchmark annotations on the figure.",
    )
    return parser.parse_args()


def read_csv(csv_path: Path) -> dict[str, dict[str, float]]:
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
    return {
        label: {
            metric: sum(item[metric] for item in rows) / len(rows)
            for metric in rows[0].keys()
        }
        for label, rows in series.items()
    }


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


def normalize(image: np.ndarray, lower: float | None = None, upper: float | None = None) -> np.ndarray:
    if lower is None:
        lower = float(np.nanpercentile(image, 2))
    if upper is None:
        upper = float(np.nanpercentile(image, 98))
    return np.clip((image - lower) / max(upper - lower, 1e-9), 0, 1)


def center_window(shape: tuple[int, int, int], crop_size: int) -> tuple[slice, slice]:
    height, width = shape[1], shape[2]
    lines = min(height, crop_size)
    samples = min(width, crop_size)
    y0 = max((height - lines) // 2, 0)
    x0 = max((width - samples) // 2, 0)
    return slice(y0, y0 + lines), slice(x0, x0 + samples)


def pca_component(cube: np.ndarray) -> np.ndarray:
    bands, height, width = cube.shape
    pixels = cube.reshape(bands, -1).T
    scores = PCA(n_components=1).fit_transform(pixels)
    return scores.reshape(height, width)


def format_seconds(value: float) -> str:
    if value >= 0.01:
        return f"{value:.4f}s"
    return f"{value:.2e}s"


def metrics_for(summary: dict[str, dict[str, float]], sizes: dict[str, float], format_key: str) -> str:
    open_s = summary[f"{format_key}_open"]["open_seconds"]
    random_s = summary[f"{format_key}_random_navigation"]["per_step_seconds"]
    pca_s = summary[f"{format_key}_pca"]["total_seconds"]
    return (
        f"size {sizes[format_key]:.1f} MiB\n"
        f"open {format_seconds(open_s)}\n"
        f"random nav {format_seconds(random_s)}/step\n"
        f"PCA {format_seconds(pca_s)}"
    )


def decorate(ax, title: str, subtitle: str, color: str) -> None:
    ax.set_facecolor(PANEL)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(color)
        spine.set_linewidth(2.2)
    ax.text(
        0.02,
        1.035,
        title,
        transform=ax.transAxes,
        color=color,
        fontsize=11,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    if subtitle:
        ax.text(
            0.02,
            -0.10,
            subtitle,
            transform=ax.transAxes,
            color=MUTED,
            fontsize=8.2,
            ha="left",
            va="top",
            linespacing=1.25,
        )


def add_badge(ax, text: str, color: str) -> None:
    ax.text(
        0.03,
        0.05,
        text,
        transform=ax.transAxes,
        color="white",
        fontsize=8.5,
        ha="left",
        va="bottom",
        bbox={"facecolor": color, "alpha": 0.88, "pad": 4, "edgecolor": "none"},
    )


def render_dataset(
    base_dir: Path,
    out_dir: Path,
    dataset_key: str,
    dataset_label: str,
    zarr_name: str,
    csv_name: str,
    report_name: str,
    band_index: int,
    crop_size: int,
    show_metrics: bool = True,
) -> Path:
    summary = read_csv(base_dir / csv_name)
    sizes = read_sizes(base_dir / report_name)
    root = zarr.open(str(base_dir / zarr_name), mode="r")
    cube = root["0"]
    band_index = min(band_index, cube.shape[0] - 1)

    band = np.asarray(cube[band_index, :, :], dtype=np.float32)
    y_slice, x_slice = center_window(cube.shape, crop_size)
    crop = np.asarray(cube[:, y_slice, x_slice], dtype=np.float32)
    pca = pca_component(crop)

    band_image = normalize(band)
    pca_image = normalize(pca)

    fig, axes = plt.subplots(
        2,
        len(FORMATS),
        figsize=(16.0, 7.8),
        facecolor=BACKGROUND,
    )
    for col, format_key in enumerate(FORMATS):
        color = COLORS[format_key]
        label = FORMAT_LABELS[format_key]
        subtitle = metrics_for(summary, sizes, format_key) if show_metrics else ""

        axes[0, col].imshow(band_image, cmap="inferno")
        decorate(axes[0, col], f"{label} Band View", subtitle, color)
        add_badge(axes[0, col], f"band {band_index + 1}", color)

        axes[1, col].imshow(pca_image, cmap="magma")
        decorate(axes[1, col], f"{label} PCA View", subtitle, color)
        add_badge(axes[1, col], "PCA component 1", color)

    fig.suptitle(
        f"{dataset_label} Benchmark Comparison Using napari-Style Views",
        fontsize=17,
        fontweight="bold",
        color="#202020",
    )
    fig.text(
        0.5,
        0.028,
        "Visual panels use the same representative OME-Zarr image data for each format because all variants encode the same cube.",
        ha="center",
        color="#303030",
        fontsize=9.5,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))

    suffix = "" if show_metrics else "_no_numbers"
    output_path = out_dir / f"{dataset_key}_napari_benchmark_view{suffix}.png"
    fig.savefig(output_path, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir
    out_dir = args.out_dir or base_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    for dataset in DATASETS:
        if args.dataset is not None and dataset[0] != args.dataset:
            continue
        outputs.append(
            render_dataset(
                base_dir,
                out_dir,
                *dataset,
                show_metrics=not args.hide_metrics,
            )
        )
    for output in outputs:
        print(f"Figure written to: {output.resolve()}")


if __name__ == "__main__":
    main()
