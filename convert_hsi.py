import argparse
import csv
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy.io
import zarr
from sklearn.decomposition import PCA

from ome_zarr.io import parse_url
from ome_zarr.scale import Scaler
from ome_zarr.writer import write_image


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


FORMAT_ORDER = ["matlab", "envi", "npy", "omezarr"]
FORMAT_LABELS = {
    "matlab": "MATLAB",
    "envi": "ENVI",
    "npy": "NumPy",
    "omezarr": "OME-Zarr",
}
SCENARIOS = {
    "open": "Initial open",
    "sequential_navigation": "Sequential wavelength navigation",
    "random_navigation": "Random wavelength navigation",
    "pca": "PCA end-to-end",
}
ENVI_DTYPES = {
    1: np.uint8,
    2: np.int16,
    3: np.int32,
    4: np.float32,
    5: np.float64,
    6: np.complex64,
    9: np.complex128,
    12: np.uint16,
    13: np.uint32,
    14: np.int64,
    15: np.uint64,
}


@dataclass
class CubeMetadata:
    shape: tuple[int, int, int]
    wavelength: np.ndarray | None = None


@dataclass
class BenchmarkSource:
    format_id: str
    label: str
    metadata: CubeMetadata
    reader_factory: callable


class CubeReader:
    shape: tuple[int, int, int]
    wavelength: np.ndarray | None

    def spectrum(self, sample_y: int, sample_x: int) -> np.ndarray:
        raise NotImplementedError

    def band(self, band_index: int) -> np.ndarray:
        raise NotImplementedError

    def region(self, y_slice: slice, x_slice: slice) -> np.ndarray:
        raise NotImplementedError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark MATLAB, ENVI, NumPy, and OME-Zarr hyperspectral access."
    )
    parser.add_argument("--mat-file", default="Beef.mat")
    parser.add_argument("--cube-var", default="Image")
    parser.add_argument("--wave-var", default="Wavelength")
    parser.add_argument("--npy-file", default=None)
    parser.add_argument("--npy-spectral-axis", type=int, choices=[0, 1, 2], default=None)
    parser.add_argument("--envi-hdr", default=None)
    parser.add_argument("--envi-dat", default=None)
    parser.add_argument("--zarr-file", default="Beef.ome.zarr")
    parser.add_argument(
        "--convert-source",
        choices=["auto", "matlab", "envi", "npy"],
        default="auto",
        help="Source format to use when creating OME-Zarr.",
    )
    parser.add_argument(
        "--reconvert",
        action="store_true",
        help="Create a fresh timestamped OME-Zarr dataset before benchmarking.",
    )
    parser.add_argument(
        "--view",
        action="store_true",
        help="Open the OME-Zarr result in napari after benchmarking.",
    )
    parser.add_argument(
        "--warmups",
        type=int,
        default=1,
        help="Number of untimed warmup runs before timed runs.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of timed runs to report.",
    )
    parser.add_argument(
        "--navigation-steps",
        type=int,
        default=100,
        help="Number of wavelength moves to benchmark for sequential and random navigation.",
    )
    parser.add_argument(
        "--csv-out",
        default="benchmark_results.csv",
        help="CSV path for per-run benchmark results.",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=3,
        help="Number of PCA components to compute.",
    )
    parser.add_argument(
        "--pca-lines",
        type=int,
        default=512,
        help="Maximum number of image lines to use for PCA materialization.",
    )
    parser.add_argument(
        "--pca-samples",
        type=int,
        default=512,
        help="Maximum number of image samples to use for PCA materialization.",
    )
    parser.add_argument(
        "--skip-pca",
        action="store_true",
        help="Skip PCA benchmarks.",
    )
    return parser.parse_args()


def resolve_output_path(zarr_file: str) -> Path:
    zarr_path = Path(zarr_file)
    if zarr_path.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zarr_path = zarr_path.with_name(f"{zarr_path.stem}_{timestamp}{zarr_path.suffix}")
    return zarr_path


def orient_cube(
    raw_cube: np.ndarray,
    wavelength: np.ndarray | None = None,
    spectral_axis: int | None = None,
    band_count_hint: int | None = None,
) -> np.ndarray:
    if raw_cube.ndim != 3:
        raise ValueError(f"Expected a 3D hyperspectral cube, got shape {raw_cube.shape}")
    if spectral_axis is None:
        spectral_len = len(wavelength) if wavelength is not None else band_count_hint
        if spectral_len is not None:
            matching_axes = [
                axis for axis, size in enumerate(raw_cube.shape) if size == spectral_len
            ]
            if len(matching_axes) == 1:
                spectral_axis = matching_axes[0]
    if spectral_axis is None:
        spectral_axis = 0
    return np.moveaxis(raw_cube, spectral_axis, 0)


def load_mat_cube(
    mat_file: str,
    cube_var: str,
    wave_var: str | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    try:
        mat = scipy.io.loadmat(mat_file)
        wavelength = np.asarray(mat[wave_var]).flatten() if wave_var is not None else None
        cube = orient_cube(np.asarray(mat[cube_var]), wavelength=wavelength)
        return cube, wavelength
    except NotImplementedError as exc:
        if "matlab v7.3" not in str(exc).lower():
            raise
        try:
            import h5py
        except ImportError as import_exc:
            raise ImportError(
                "MATLAB v7.3 files require h5py. Install it in the project venv."
            ) from import_exc

        with h5py.File(mat_file, "r") as mat:
            wavelength = (
                np.asarray(mat[wave_var]).flatten() if wave_var is not None else None
            )
            cube = orient_cube(np.asarray(mat[cube_var]), wavelength=wavelength)
        return cube, wavelength


def inspect_mat_metadata(
    mat_file: str,
    cube_var: str,
    wave_var: str | None = None,
) -> CubeMetadata:
    try:
        mat = scipy.io.loadmat(mat_file)
        wavelength = np.asarray(mat[wave_var]).flatten() if wave_var is not None else None
        cube = orient_cube(np.asarray(mat[cube_var]), wavelength=wavelength)
        return CubeMetadata(shape=cube.shape, wavelength=wavelength)
    except NotImplementedError as exc:
        if "matlab v7.3" not in str(exc).lower():
            raise
        import h5py

        with h5py.File(mat_file, "r") as mat:
            wavelength = (
                np.asarray(mat[wave_var]).flatten() if wave_var is not None else None
            )
            raw_shape = tuple(int(size) for size in mat[cube_var].shape)
        cube = orient_cube(
            np.empty(raw_shape, dtype=np.uint8),
            wavelength=wavelength,
        )
        return CubeMetadata(shape=cube.shape, wavelength=wavelength)


def parse_envi_header(hdr_path: Path) -> dict[str, str]:
    text = hdr_path.read_text()
    fields: dict[str, str] = {}
    key: str | None = None
    buffer: list[str] = []
    depth = 0
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line == "ENVI":
            continue
        if key is None:
            if "=" not in line:
                continue
            maybe_key, maybe_value = [part.strip() for part in line.split("=", 1)]
            key = maybe_key.lower()
            buffer = [maybe_value]
            depth = maybe_value.count("{") - maybe_value.count("}")
            if depth <= 0:
                fields[key] = " ".join(buffer).strip()
                key = None
        else:
            buffer.append(line)
            depth += line.count("{") - line.count("}")
            if depth <= 0:
                fields[key] = " ".join(buffer).strip()
                key = None
    return fields


def parse_envi_list(value: str) -> list[float]:
    cleaned = value.strip()
    if cleaned.startswith("{") and cleaned.endswith("}"):
        cleaned = cleaned[1:-1]
    return [float(item.strip()) for item in cleaned.split(",") if item.strip()]


def envi_byteorder_dtype(base_dtype: np.dtype, byte_order: int) -> np.dtype:
    if base_dtype.kind in {"u", "i", "f", "c"}:
        return base_dtype.newbyteorder("<" if byte_order == 0 else ">")
    return base_dtype


def default_envi_dat_path(hdr_path: Path) -> Path:
    direct = hdr_path.with_suffix(".dat")
    if direct.exists():
        return direct
    return hdr_path.with_suffix("")


class MatlabReader(CubeReader):
    def __init__(self, mat_file: str, cube_var: str, wave_var: str | None):
        self.cube, self.wavelength = load_mat_cube(mat_file, cube_var, wave_var)
        self.shape = self.cube.shape

    def spectrum(self, sample_y: int, sample_x: int) -> np.ndarray:
        return self.cube[:, sample_y, sample_x]

    def band(self, band_index: int) -> np.ndarray:
        return self.cube[band_index, :, :]

    def region(self, y_slice: slice, x_slice: slice) -> np.ndarray:
        return self.cube[:, y_slice, x_slice]


class NpyReader(CubeReader):
    def __init__(
        self,
        npy_file: str,
        spectral_axis: int | None,
        band_count_hint: int | None,
    ):
        raw_cube = np.load(npy_file, mmap_mode="r")
        self.cube = orient_cube(
            raw_cube,
            spectral_axis=spectral_axis,
            band_count_hint=band_count_hint,
        )
        self.wavelength = None
        self.shape = tuple(int(size) for size in self.cube.shape)

    def spectrum(self, sample_y: int, sample_x: int) -> np.ndarray:
        return np.asarray(self.cube[:, sample_y, sample_x])

    def band(self, band_index: int) -> np.ndarray:
        return np.asarray(self.cube[band_index, :, :])

    def region(self, y_slice: slice, x_slice: slice) -> np.ndarray:
        return np.asarray(self.cube[:, y_slice, x_slice])


class EnviReader(CubeReader):
    def __init__(
        self,
        hdr_path: str,
        dat_path: str | None = None,
        transpose_spatial: bool = False,
    ):
        self.hdr_path = Path(hdr_path)
        self.transpose_spatial = transpose_spatial
        header = parse_envi_header(self.hdr_path)
        self.lines = int(header["lines"])
        self.samples = int(header["samples"])
        self.bands = int(header["bands"])
        self.interleave = header["interleave"].strip().lower()
        data_type = int(header["data type"])
        if data_type not in ENVI_DTYPES:
            raise ValueError(f"Unsupported ENVI data type: {data_type}")
        byte_order = int(header.get("byte order", "0"))
        dtype = envi_byteorder_dtype(np.dtype(ENVI_DTYPES[data_type]), byte_order)
        offset = int(header.get("header offset", "0"))
        wavelength_value = header.get("wavelength")
        self.wavelength = (
            np.asarray(parse_envi_list(wavelength_value), dtype=np.float64)
            if wavelength_value is not None
            else None
        )
        dat_file = Path(dat_path) if dat_path is not None else default_envi_dat_path(self.hdr_path)
        if self.interleave == "bil":
            base = np.memmap(
                dat_file,
                mode="r",
                dtype=dtype,
                shape=(self.lines, self.bands, self.samples),
                offset=offset,
            )
            self.cube = np.transpose(base, (1, 0, 2))
        elif self.interleave == "bip":
            base = np.memmap(
                dat_file,
                mode="r",
                dtype=dtype,
                shape=(self.lines, self.samples, self.bands),
                offset=offset,
            )
            self.cube = np.transpose(base, (2, 0, 1))
        elif self.interleave == "bsq":
            self.cube = np.memmap(
                dat_file,
                mode="r",
                dtype=dtype,
                shape=(self.bands, self.lines, self.samples),
                offset=offset,
            )
        else:
            raise ValueError(f"Unsupported ENVI interleave: {self.interleave}")
        base_shape = tuple(int(size) for size in self.cube.shape)
        if self.transpose_spatial:
            self.shape = (base_shape[0], base_shape[2], base_shape[1])
        else:
            self.shape = base_shape

    def spectrum(self, sample_y: int, sample_x: int) -> np.ndarray:
        if self.transpose_spatial:
            return np.asarray(self.cube[:, sample_x, sample_y])
        return np.asarray(self.cube[:, sample_y, sample_x])

    def band(self, band_index: int) -> np.ndarray:
        band = np.asarray(self.cube[band_index, :, :])
        if self.transpose_spatial:
            return band.T
        return band

    def region(self, y_slice: slice, x_slice: slice) -> np.ndarray:
        if self.transpose_spatial:
            return np.asarray(self.cube[:, x_slice, y_slice]).transpose(0, 2, 1)
        return np.asarray(self.cube[:, y_slice, x_slice])


class OmezarrReader(CubeReader):
    def __init__(self, zarr_path: str):
        root = zarr.open(zarr_path, mode="r")
        self.cube = root["0"]
        wavelength = root.attrs.get("wavelength_nm")
        self.wavelength = (
            np.asarray(wavelength, dtype=np.float64) if wavelength is not None else None
        )
        self.shape = tuple(int(size) for size in self.cube.shape)

    def spectrum(self, sample_y: int, sample_x: int) -> np.ndarray:
        return np.asarray(self.cube[:, sample_y, sample_x])

    def band(self, band_index: int) -> np.ndarray:
        return np.asarray(self.cube[band_index, :, :])

    def region(self, y_slice: slice, x_slice: slice) -> np.ndarray:
        return np.asarray(self.cube[:, y_slice, x_slice])


def source_exists(path: str | None) -> bool:
    return path is not None and Path(path).exists()


def first_available_band_count(args: argparse.Namespace) -> int | None:
    if source_exists(args.envi_hdr):
        return int(parse_envi_header(Path(args.envi_hdr))["bands"])
    if source_exists(args.mat_file):
        return inspect_mat_metadata(args.mat_file, args.cube_var, args.wave_var).shape[0]
    if source_exists(args.zarr_file):
        return int(zarr.open(args.zarr_file, mode="r")["0"].shape[0])
    if source_exists(args.npy_file):
        raw = np.load(args.npy_file, mmap_mode="r")
        if args.npy_spectral_axis is not None:
            return int(raw.shape[args.npy_spectral_axis])
    return None


def build_sources(args: argparse.Namespace) -> list[BenchmarkSource]:
    band_count_hint = first_available_band_count(args)
    sources: list[BenchmarkSource] = []
    reference_shape: tuple[int, int, int] | None = None

    if source_exists(args.mat_file):
        metadata = inspect_mat_metadata(args.mat_file, args.cube_var, args.wave_var)
        reference_shape = metadata.shape
        sources.append(
            BenchmarkSource(
                format_id="matlab",
                label=FORMAT_LABELS["matlab"],
                metadata=metadata,
                reader_factory=lambda: MatlabReader(
                    args.mat_file,
                    args.cube_var,
                    args.wave_var,
                ),
            )
        )

    if source_exists(args.envi_hdr):
        envi_reader = EnviReader(args.envi_hdr, args.envi_dat)
        transpose_spatial = False
        if (
            reference_shape is not None
            and envi_reader.shape[0] == reference_shape[0]
            and envi_reader.shape[1] == reference_shape[2]
            and envi_reader.shape[2] == reference_shape[1]
        ):
            transpose_spatial = True
            envi_reader = EnviReader(args.envi_hdr, args.envi_dat, transpose_spatial=True)
        metadata = CubeMetadata(shape=envi_reader.shape, wavelength=envi_reader.wavelength)
        sources.append(
            BenchmarkSource(
                format_id="envi",
                label=FORMAT_LABELS["envi"],
                metadata=metadata,
                reader_factory=lambda: EnviReader(
                    args.envi_hdr,
                    args.envi_dat,
                    transpose_spatial=transpose_spatial,
                ),
            )
        )

    if source_exists(args.npy_file):
        npy_reader = NpyReader(args.npy_file, args.npy_spectral_axis, band_count_hint)
        metadata = CubeMetadata(shape=npy_reader.shape, wavelength=npy_reader.wavelength)
        sources.append(
            BenchmarkSource(
                format_id="npy",
                label=FORMAT_LABELS["npy"],
                metadata=metadata,
                reader_factory=lambda: NpyReader(
                    args.npy_file,
                    args.npy_spectral_axis,
                    band_count_hint,
                ),
            )
        )

    if source_exists(args.zarr_file):
        zarr_reader = OmezarrReader(args.zarr_file)
        metadata = CubeMetadata(shape=zarr_reader.shape, wavelength=zarr_reader.wavelength)
        sources.append(
            BenchmarkSource(
                format_id="omezarr",
                label=FORMAT_LABELS["omezarr"],
                metadata=metadata,
                reader_factory=lambda: OmezarrReader(args.zarr_file),
            )
        )

    sources.sort(key=lambda item: FORMAT_ORDER.index(item.format_id))
    if not sources:
        raise FileNotFoundError("No benchmarkable input files were found.")
    return sources


def validate_shapes(sources: list[BenchmarkSource]) -> None:
    reference = sources[0].metadata.shape
    for source in sources[1:]:
        if source.metadata.shape != reference:
            raise ValueError(
                f"Shape mismatch: {source.label} has {source.metadata.shape}, expected {reference}"
            )


def choose_conversion_source(
    args: argparse.Namespace,
    sources: list[BenchmarkSource],
) -> BenchmarkSource:
    if args.convert_source != "auto":
        for source in sources:
            if source.format_id == args.convert_source:
                return source
        raise ValueError(f"Requested conversion source '{args.convert_source}' is unavailable.")
    for format_id in ["matlab", "envi", "npy"]:
        for source in sources:
            if source.format_id == format_id:
                return source
    raise ValueError("No source available to create OME-Zarr.")


def convert_to_omezarr(
    cube: np.ndarray,
    wavelength: np.ndarray | None,
    zarr_path: Path,
) -> tuple[Path, float]:
    t0 = perf_counter()
    store = parse_url(str(zarr_path), mode="w").store
    root = zarr.group(store=store)
    write_image(
        image=cube,
        group=root,
        axes="cyx",
        scaler=Scaler(max_layer=4),
    )
    if wavelength is not None:
        root.attrs["wavelength_nm"] = wavelength.tolist()
    return zarr_path, perf_counter() - t0


def ensure_omezarr(args: argparse.Namespace, sources: list[BenchmarkSource]) -> list[BenchmarkSource]:
    zarr_path = Path(args.zarr_file)
    if args.reconvert:
        zarr_path = resolve_output_path(args.zarr_file)
        args.zarr_file = str(zarr_path)

    if args.reconvert or not zarr_path.exists():
        source = choose_conversion_source(args, sources)
        reader = source.reader_factory()
        cube = np.asarray(reader.region(slice(0, reader.shape[1]), slice(0, reader.shape[2])))
        wavelength = reader.wavelength
        print(f"Converting {source.label} to OME-Zarr: {zarr_path}")
        _, convert_seconds = convert_to_omezarr(cube, wavelength, zarr_path)
        print(f"Conversion time: {convert_seconds:.4f}s")

    return build_sources(args)


def center_pixel(shape: tuple[int, ...]) -> tuple[int, int]:
    return shape[1] // 2, shape[2] // 2


def pca_window(shape: tuple[int, int, int], max_lines: int, max_samples: int) -> tuple[slice, slice]:
    line_count = min(shape[1], max_lines)
    sample_count = min(shape[2], max_samples)
    start_y = max((shape[1] - line_count) // 2, 0)
    start_x = max((shape[2] - sample_count) // 2, 0)
    return slice(start_y, start_y + line_count), slice(start_x, start_x + sample_count)


def benchmark_open(source: BenchmarkSource, sample_y: int, sample_x: int) -> dict[str, float]:
    t0 = perf_counter()
    reader = source.reader_factory()
    t1 = perf_counter()
    _ = reader.spectrum(sample_y, sample_x)
    t2 = perf_counter()
    _ = reader.band(0)
    t3 = perf_counter()
    return {
        "open_seconds": t1 - t0,
        "spectrum_seconds": t2 - t1,
        "slice_seconds": t3 - t2,
        "total_seconds": t3 - t0,
    }


def benchmark_navigation(
    source: BenchmarkSource,
    band_indices: list[int],
) -> dict[str, float]:
    reader = source.reader_factory()
    t0 = perf_counter()
    for band in band_indices:
        _ = reader.band(band)
    t1 = perf_counter()
    return {
        "steps": float(len(band_indices)),
        "total_seconds": t1 - t0,
        "per_step_seconds": (t1 - t0) / len(band_indices),
    }


def compute_pca_scores(cube: np.ndarray, components: int) -> tuple[np.ndarray, float]:
    bands, height, width = cube.shape
    pixels_by_band = cube.reshape(bands, -1).T
    t0 = perf_counter()
    scores = PCA(n_components=components).fit_transform(pixels_by_band)
    t1 = perf_counter()
    return scores.T.reshape(components, height, width), t1 - t0


def benchmark_pca(
    source: BenchmarkSource,
    components: int,
    y_slice: slice,
    x_slice: slice,
) -> dict[str, float]:
    t0 = perf_counter()
    reader = source.reader_factory()
    t1 = perf_counter()
    cube = np.asarray(reader.region(y_slice, x_slice))
    t2 = perf_counter()
    _, pca_seconds = compute_pca_scores(cube, components)
    t3 = perf_counter()
    return {
        "open_seconds": t1 - t0,
        "materialize_seconds": t2 - t1,
        "pca_seconds": pca_seconds,
        "total_seconds": t3 - t0,
    }


def sequential_bands(band_count: int, steps: int) -> list[int]:
    return [index % band_count for index in range(steps)]


def random_bands(band_count: int, steps: int) -> list[int]:
    rng = random.Random(42)
    return [rng.randrange(band_count) for _ in range(steps)]


def run_series(label: str, func, warmups: int, runs: int) -> list[dict[str, float]]:
    for _ in range(warmups):
        func()
    results = []
    for index in range(runs):
        result = func()
        results.append(result)
        print(f"{label} run {index + 1}: {result}")
    return results


def summarize(label: str, results: list[dict[str, float]]) -> None:
    print(f"\n{label} summary")
    keys = results[0].keys()
    for key in keys:
        values = [result[key] for result in results]
        average = sum(values) / len(values)
        minimum = min(values)
        maximum = max(values)
        print(f"  {key}: avg={average:.4f}s min={minimum:.4f}s max={maximum:.4f}s")


def averages(results: list[dict[str, float]]) -> dict[str, float]:
    keys = results[0].keys()
    return {key: sum(result[key] for result in results) / len(results) for key in keys}


def print_comparison_summary(
    results_by_series: dict[str, list[dict[str, float]]],
    formats: list[str],
    include_pca: bool,
) -> None:
    print("\nComparison summary")
    comparisons = [
        ("open", "open_seconds"),
        ("sequential_navigation", "per_step_seconds"),
        ("random_navigation", "per_step_seconds"),
    ]
    if include_pca:
        comparisons.append(("pca", "total_seconds"))

    for scenario, metric in comparisons:
        ranking = []
        for format_id in formats:
            series_name = f"{format_id}_{scenario}"
            if series_name not in results_by_series:
                continue
            average_value = averages(results_by_series[series_name])[metric]
            ranking.append((average_value, FORMAT_LABELS[format_id]))
        ranking.sort(key=lambda item: item[0])
        if not ranking:
            continue
        print(f"  {SCENARIOS[scenario]}:")
        for value, label in ranking:
            print(f"    {label}: {value:.4f}s")


def write_csv(
    csv_path: Path,
    series: dict[str, list[dict[str, float]]],
) -> None:
    fieldnames = ["series", "run"]
    metric_names = sorted(
        {
            metric
            for results in series.values()
            for result in results
            for metric in result.keys()
        }
    )
    fieldnames.extend(metric_names)

    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for label, results in series.items():
            for index, result in enumerate(results, start=1):
                row = {"series": label, "run": index}
                row.update(result)
                writer.writerow(row)


def open_in_napari(
    zarr_path: Path,
    wavelength: np.ndarray | None,
    pca_images: np.ndarray | None = None,
) -> None:
    import matplotlib.pyplot as plt
    import napari

    cube_zarr = zarr.open(str(zarr_path), mode="r")["0"]
    viewer = napari.Viewer(show=True)
    viewer.window._qt_window.show()
    viewer.window._qt_window.raise_()
    viewer.window._qt_window.activateWindow()
    layer = viewer.add_image(cube_zarr, name=zarr_path.name)

    def plot_spectrum(layer, event) -> None:
        if wavelength is None:
            return
        coords = layer.coordinates.astype(int)
        _, y, x = coords
        spectrum = cube_zarr[:, y, x]
        plt.figure()
        plt.plot(wavelength, spectrum)
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Intensity")
        plt.title(f"Spectrum at pixel ({y}, {x})")
        plt.show()

    layer.mouse_drag_callbacks.append(plot_spectrum)
    if pca_images is not None:
        viewer.add_image(
            pca_images,
            name="PCA components",
            channel_axis=0,
        )
    print("\nViewer ready.")
    print("Click any pixel to display its spectrum.")
    napari.run()


def main() -> None:
    configure_runtime()
    args = parse_args()

    sources = build_sources(args)
    sources = ensure_omezarr(args, sources)
    validate_shapes(sources)
    formats = [source.format_id for source in sources]
    reference = sources[0]
    cube_shape = reference.metadata.shape
    wavelength = reference.metadata.wavelength
    print("Available formats:", ", ".join(source.label for source in sources))
    print("Cube shape (bands, y, x):", cube_shape)

    sample_y, sample_x = center_pixel(cube_shape)
    print(f"Sample pixel: y={sample_y}, x={sample_x}")
    sequential_indices = sequential_bands(cube_shape[0], args.navigation_steps)
    random_indices = random_bands(cube_shape[0], args.navigation_steps)
    y_slice, x_slice = pca_window(cube_shape, args.pca_lines, args.pca_samples)
    print(
        "PCA window:",
        f"bands={cube_shape[0]}, lines={y_slice.stop - y_slice.start}, samples={x_slice.stop - x_slice.start}",
    )

    results_by_series: dict[str, list[dict[str, float]]] = {}
    for source in sources:
        results_by_series[f"{source.format_id}_open"] = run_series(
            f"{source.label} open",
            lambda source=source: benchmark_open(source, sample_y, sample_x),
            args.warmups,
            args.runs,
        )

    for source in sources:
        results_by_series[f"{source.format_id}_sequential_navigation"] = run_series(
            f"{source.label} sequential navigation",
            lambda source=source: benchmark_navigation(source, sequential_indices),
            args.warmups,
            args.runs,
        )
        results_by_series[f"{source.format_id}_random_navigation"] = run_series(
            f"{source.label} random navigation",
            lambda source=source: benchmark_navigation(source, random_indices),
            args.warmups,
            args.runs,
        )

    if not args.skip_pca:
        for source in sources:
            results_by_series[f"{source.format_id}_pca"] = run_series(
                f"{source.label} PCA",
                lambda source=source: benchmark_pca(
                    source,
                    args.pca_components,
                    y_slice,
                    x_slice,
                ),
                args.warmups,
                args.runs,
            )

    for series_name, results in results_by_series.items():
        summarize(series_name, results)
    print_comparison_summary(results_by_series, formats, include_pca=not args.skip_pca)

    csv_path = Path(args.csv_out)
    write_csv(csv_path, results_by_series)
    print(f"\nCSV written to: {csv_path.resolve()}")

    if args.view:
        pca_images = None
        if not args.skip_pca:
            omezarr_source = next(
                (source for source in sources if source.format_id == "omezarr"),
                None,
            )
            if omezarr_source is not None:
                cube = np.asarray(omezarr_source.reader_factory().region(y_slice, x_slice))
                pca_images, _ = compute_pca_scores(cube, args.pca_components)
        open_in_napari(Path(args.zarr_file), wavelength, pca_images)


if __name__ == "__main__":
    main()
