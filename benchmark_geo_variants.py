import argparse
import csv
import importlib.util
from dataclasses import dataclass
from pathlib import Path


def load_convert_hsi_module():
    script_path = Path(__file__).with_name("convert_hsi.py")
    spec = importlib.util.spec_from_file_location("convert_hsi", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


CONVERT_HSI = load_convert_hsi_module()


@dataclass
class Source:
    key: str
    label: str
    size_bytes: int
    metadata: object
    reader_factory: object


def parse_args() -> argparse.Namespace:
    base = Path(__file__).parent
    parser = argparse.ArgumentParser(
        description="Benchmark Geo dataset across MAT, ENVI, NumPy, and OME-Zarr variants."
    )
    parser.add_argument("--mat-file", default=str(base / "Geo.mat"))
    parser.add_argument("--mat-uncompressed-file", default=str(base / "Geo_uncompressed.mat"))
    parser.add_argument("--cube-var", default="Image")
    parser.add_argument("--wave-var", default="Wavelength")
    parser.add_argument("--envi-hdr", default=str(base / "Geo.hdr"))
    parser.add_argument("--envi-dat", default=str(base / "Geo.dat"))
    parser.add_argument("--npy-file", default=str(base / "Geo_Image.npy"))
    parser.add_argument("--npy-spectral-axis", type=int, default=None)
    parser.add_argument("--zarr-file", default=str(base / "Geo.ome.zarr"))
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--navigation-steps", type=int, default=20)
    parser.add_argument("--pca-lines", type=int, default=128)
    parser.add_argument("--pca-samples", type=int, default=128)
    parser.add_argument("--pca-components", type=int, default=3)
    parser.add_argument("--csv-out", default=str(base / "geo_variants_benchmark.csv"))
    parser.add_argument("--report-out", default=str(base / "geo_variants_report.md"))
    return parser.parse_args()


def file_size_bytes(path: str) -> int:
    p = Path(path)
    if p.is_dir():
        return sum(item.stat().st_size for item in p.rglob("*") if item.is_file())
    return p.stat().st_size


def format_size(num_bytes: int) -> str:
    return f"{num_bytes / 1024 / 1024:.1f} MiB"


def averages(results: list[dict[str, float]]) -> dict[str, float]:
    return CONVERT_HSI.averages(results)


def read_csv(csv_path: Path) -> dict[str, list[dict[str, float]]]:
    series: dict[str, list[dict[str, float]]] = {}
    with csv_path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            label = row["series"]
            metrics: dict[str, float] = {}
            for key, value in row.items():
                if key in {"series", "run"} or value in {"", None}:
                    continue
                metrics[key] = float(value)
            series.setdefault(label, []).append(metrics)
    return series


def build_sources(args: argparse.Namespace) -> list[Source]:
    mat_meta = CONVERT_HSI.inspect_mat_metadata(args.mat_file, args.cube_var, args.wave_var)
    mat_un_meta = CONVERT_HSI.inspect_mat_metadata(
        args.mat_uncompressed_file, args.cube_var, args.wave_var
    )
    envi_reader = CONVERT_HSI.EnviReader(args.envi_hdr, args.envi_dat, transpose_spatial=True)
    npy_reader = CONVERT_HSI.NpyReader(args.npy_file, args.npy_spectral_axis, mat_meta.shape[0])
    zarr_reader = CONVERT_HSI.OmezarrReader(args.zarr_file)

    sources = [
        Source(
            "matlab_compressed",
            "MATLAB compressed",
            file_size_bytes(args.mat_file),
            mat_meta,
            lambda: CONVERT_HSI.MatlabReader(args.mat_file, args.cube_var, args.wave_var),
        ),
        Source(
            "matlab_uncompressed",
            "MATLAB uncompressed",
            file_size_bytes(args.mat_uncompressed_file),
            mat_un_meta,
            lambda: CONVERT_HSI.MatlabReader(
                args.mat_uncompressed_file, args.cube_var, args.wave_var
            ),
        ),
        Source(
            "envi",
            "ENVI",
            file_size_bytes(args.envi_hdr) + file_size_bytes(args.envi_dat),
            CONVERT_HSI.CubeMetadata(envi_reader.shape, envi_reader.wavelength),
            lambda: CONVERT_HSI.EnviReader(args.envi_hdr, args.envi_dat, transpose_spatial=True),
        ),
        Source(
            "npy",
            "NumPy",
            file_size_bytes(args.npy_file),
            CONVERT_HSI.CubeMetadata(npy_reader.shape, npy_reader.wavelength),
            lambda: CONVERT_HSI.NpyReader(args.npy_file, args.npy_spectral_axis, mat_meta.shape[0]),
        ),
        Source(
            "omezarr",
            "OME-Zarr",
            file_size_bytes(args.zarr_file),
            CONVERT_HSI.CubeMetadata(zarr_reader.shape, zarr_reader.wavelength),
            lambda: CONVERT_HSI.OmezarrReader(args.zarr_file),
        ),
    ]
    reference = sources[0].metadata.shape
    for source in sources[1:]:
        if source.metadata.shape != reference:
            raise ValueError(
                f"Shape mismatch: {source.label} has {source.metadata.shape}, expected {reference}"
            )
    return sources


def write_csv(csv_path: Path, series: dict[str, list[dict[str, float]]]) -> None:
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


def generate_report(args: argparse.Namespace, sources: list[Source], csv_path: Path, report_path: Path) -> None:
    series = read_csv(csv_path)
    lines = [
        "# Geo Variants Comparison",
        "",
        "## Method",
        "",
        f"- Runs: `{args.runs}`",
        f"- Warmups: `{args.warmups}`",
        f"- Navigation steps: `{args.navigation_steps}`",
        f"- PCA crop: `{args.pca_lines} x {args.pca_samples}` across all bands",
        "",
        "## Summary Table",
        "",
        "| Format | File size | Initial open | Sequential navigation | Random navigation | End-to-end PCA |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for source in sources:
        open_avg = averages(series[f"{source.key}_open"])
        seq_avg = averages(series[f"{source.key}_sequential_navigation"])
        rand_avg = averages(series[f"{source.key}_random_navigation"])
        pca_avg = averages(series[f"{source.key}_pca"])
        lines.append(
            f"| {source.label} | {format_size(source.size_bytes)} | "
            f"{open_avg['open_seconds']:.4f} s | "
            f"{seq_avg['per_step_seconds']:.8f} s/step | "
            f"{rand_avg['per_step_seconds']:.8f} s/step | "
            f"{pca_avg['total_seconds']:.4f} s |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Compressed and uncompressed MATLAB store the same float64 data.",
            "- ENVI stores Geo as uint16, while the MAT, NPY, and OME-Zarr variants here are float64.",
            "- CSV: "
            f"[{csv_path.name}]({csv_path.resolve()}:1)",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    CONVERT_HSI.configure_runtime()
    sources = build_sources(args)
    shape = sources[0].metadata.shape
    sample_y, sample_x = CONVERT_HSI.center_pixel(shape)
    sequential_indices = CONVERT_HSI.sequential_bands(shape[0], args.navigation_steps)
    random_indices = CONVERT_HSI.random_bands(shape[0], args.navigation_steps)
    y_slice, x_slice = CONVERT_HSI.pca_window(shape, args.pca_lines, args.pca_samples)

    results_by_series: dict[str, list[dict[str, float]]] = {}
    for source in sources:
        results_by_series[f"{source.key}_open"] = CONVERT_HSI.run_series(
            f"{source.label} open",
            lambda source=source: CONVERT_HSI.benchmark_open(source, sample_y, sample_x),
            args.warmups,
            args.runs,
        )
        results_by_series[f"{source.key}_sequential_navigation"] = CONVERT_HSI.run_series(
            f"{source.label} sequential navigation",
            lambda source=source: CONVERT_HSI.benchmark_navigation(source, sequential_indices),
            args.warmups,
            args.runs,
        )
        results_by_series[f"{source.key}_random_navigation"] = CONVERT_HSI.run_series(
            f"{source.label} random navigation",
            lambda source=source: CONVERT_HSI.benchmark_navigation(source, random_indices),
            args.warmups,
            args.runs,
        )
        results_by_series[f"{source.key}_pca"] = CONVERT_HSI.run_series(
            f"{source.label} PCA",
            lambda source=source: CONVERT_HSI.benchmark_pca(
                source,
                args.pca_components,
                y_slice,
                x_slice,
            ),
            args.warmups,
            args.runs,
        )

    csv_path = Path(args.csv_out)
    write_csv(csv_path, results_by_series)
    report_path = Path(args.report_out)
    generate_report(args, sources, csv_path, report_path)
    print(f"CSV written to: {csv_path}")
    print(f"Report written to: {report_path}")


if __name__ == "__main__":
    main()
