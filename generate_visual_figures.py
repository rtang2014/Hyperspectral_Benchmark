import argparse
import os
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
    ("canola", "Canola", "canola.ome.zarr", 128),
    ("beef", "Beef", "Beef_fixed.ome.zarr", 128),
    ("geo", "Geo", "Geo.ome.zarr", 64),
]
BACKGROUND = "#f4f0e8"
PANEL = "#fbfaf7"
GRID_COLOR = "#d7d7d0"
TEXT_COLOR = "#242424"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate data-derived visual figures for the hyperspectral paper."
    )
    parser.add_argument(
        "--base-dir",
        default=Path(__file__).parent,
        type=Path,
        help="Directory containing OME-Zarr datasets.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        type=Path,
        help="Directory for generated figures. Defaults to base-dir.",
    )
    return parser.parse_args()


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": BACKGROUND,
            "axes.facecolor": PANEL,
            "axes.edgecolor": "#404040",
            "font.size": 10,
            "savefig.facecolor": BACKGROUND,
        }
    )


def save_figure(fig, output_path: Path) -> None:
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path.resolve()}")


def normalize(image: np.ndarray, lower: float | None = None, upper: float | None = None) -> np.ndarray:
    if lower is None:
        lower = float(np.nanpercentile(image, 2))
    if upper is None:
        upper = float(np.nanpercentile(image, 98))
    return np.clip((image - lower) / max(upper - lower, 1e-9), 0, 1)


def open_dataset(base_dir: Path, zarr_name: str):
    root = zarr.open(str(base_dir / zarr_name), mode="r")
    return root, root["0"]


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


def choose_bands(band_count: int) -> list[int]:
    return [
        min(int(round(band_count * 0.20)), band_count - 1),
        min(int(round(band_count * 0.50)), band_count - 1),
        min(int(round(band_count * 0.80)), band_count - 1),
    ]


def false_color(arr) -> np.ndarray:
    bands = choose_bands(arr.shape[0])
    channels = [np.asarray(arr[index, :, :], dtype=np.float32) for index in bands]
    normalized = [normalize(channel) for channel in channels]
    return np.dstack(normalized[::-1])


def center_window(shape: tuple[int, int, int], size: int) -> tuple[slice, slice]:
    height, width = shape[1], shape[2]
    lines = min(height, size)
    samples = min(width, size)
    y0 = max((height - lines) // 2, 0)
    x0 = max((width - samples) // 2, 0)
    return slice(y0, y0 + lines), slice(x0, x0 + samples)


def pca_rgb(cube: np.ndarray) -> np.ndarray:
    bands, height, width = cube.shape
    pixels = cube.reshape(bands, -1).T
    scores = PCA(n_components=3).fit_transform(pixels)
    components = scores.T.reshape(3, height, width)
    channels = [normalize(component) for component in components]
    return np.dstack(channels)


def wavelengths_or_indices(root, band_count: int) -> tuple[np.ndarray, str]:
    wavelengths = root.attrs.get("wavelength_nm")
    if wavelengths is not None and len(wavelengths) == band_count:
        return np.asarray(wavelengths, dtype=np.float64), "Wavelength (nm)"
    return np.arange(1, band_count + 1), "Band index"


def spectrum_positions(shape: tuple[int, int, int]) -> list[tuple[str, int, int]]:
    _, height, width = shape
    return [
        ("upper-left", height // 4, width // 4),
        ("center", height // 2, width // 2),
        ("lower-right", (height * 3) // 4, (width * 3) // 4),
    ]


def plot_dataset_overview(base_dir: Path, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(13.5, 4.5))
    fig.suptitle("Representative False-Color Views of Benchmark Datasets", fontsize=15, weight="bold")
    for ax, (_, label, zarr_name, _) in zip(axes, DATASETS):
        root, arr0 = open_dataset(base_dir, zarr_name)
        image = false_color(display_level(root))
        ax.imshow(image, aspect="auto")
        ax.set_title(f"{label}\n{arr0.shape[0]} bands, {arr0.shape[1]} x {arr0.shape[2]} pixels")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.text(
        0.5,
        0.02,
        "Composites use three representative spectral bands and percentile contrast stretching; colors are not true-color renderings.",
        ha="center",
        fontsize=9,
        color=TEXT_COLOR,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    save_figure(fig, out_dir / "visual_figure_1_dataset_overview.png")


def plot_spectra(base_dir: Path, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(13.8, 4.2))
    fig.suptitle("Representative Pixel Spectra", fontsize=15, weight="bold")
    for ax, (_, label, zarr_name, _) in zip(axes, DATASETS):
        root, arr = open_dataset(base_dir, zarr_name)
        x_values, xlabel = wavelengths_or_indices(root, arr.shape[0])
        for position_label, y, x in spectrum_positions(arr.shape):
            spectrum = np.asarray(arr[:, y, x], dtype=np.float64)
            ax.plot(x_values, spectrum, linewidth=1.4, label=position_label)
        ax.set_title(label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Intensity")
        ax.grid(True, color=GRID_COLOR, linewidth=0.8)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_figure(fig, out_dir / "visual_figure_2_representative_spectra.png")


def plot_pca_maps(base_dir: Path, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(13.5, 4.5))
    fig.suptitle("PCA Composite Views from Centered Benchmark Crops", fontsize=15, weight="bold")
    for ax, (_, label, zarr_name, crop_size) in zip(axes, DATASETS):
        _, arr = open_dataset(base_dir, zarr_name)
        y_slice, x_slice = center_window(arr.shape, crop_size)
        crop = np.asarray(arr[:, y_slice, x_slice], dtype=np.float32)
        image = pca_rgb(crop)
        ax.imshow(image)
        ax.set_title(f"{label}\n{crop.shape[1]} x {crop.shape[2]} crop, all bands")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.text(
        0.5,
        0.02,
        "RGB channels correspond to the first three PCA score images after independent percentile contrast stretching.",
        ha="center",
        fontsize=9,
        color=TEXT_COLOR,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    save_figure(fig, out_dir / "visual_figure_3_pca_composites.png")


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir
    out_dir = args.out_dir or base_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_style()
    plot_dataset_overview(base_dir, out_dir)
    plot_spectra(base_dir, out_dir)
    plot_pca_maps(base_dir, out_dir)


if __name__ == "__main__":
    main()
