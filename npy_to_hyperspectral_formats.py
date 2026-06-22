import argparse
import os
from pathlib import Path

import numpy as np
import scipy.io
import zarr

from ome_zarr.io import parse_url
from ome_zarr.scale import Scaler
from ome_zarr.writer import write_image


ENVI_DTYPES = {
    np.dtype("uint8"): 1,
    np.dtype("int16"): 2,
    np.dtype("int32"): 3,
    np.dtype("float32"): 4,
    np.dtype("float64"): 5,
    np.dtype("complex64"): 6,
    np.dtype("complex128"): 9,
    np.dtype("uint16"): 12,
    np.dtype("uint32"): 13,
    np.dtype("int64"): 14,
    np.dtype("uint64"): 15,
}


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a hyperspectral .npy array into MAT, ENVI, and OME-Zarr variants."
    )
    parser.add_argument("npy_file", help="Input .npy file")
    parser.add_argument(
        "--spectral-axis",
        type=int,
        choices=[0, 1, 2],
        default=2,
        help="Band axis in the source .npy array. Default assumes (y, x, bands).",
    )
    parser.add_argument(
        "--mat-var",
        default="Image",
        help="Variable name to store in MAT files.",
    )
    parser.add_argument(
        "--output-stem",
        help="Base output path without extension. Defaults to the input stem in the same folder.",
    )
    return parser.parse_args()


def move_spectral_last(array: np.ndarray, spectral_axis: int) -> np.ndarray:
    return np.moveaxis(array, spectral_axis, -1)


def move_spectral_first(array: np.ndarray, spectral_axis: int) -> np.ndarray:
    return np.moveaxis(array, spectral_axis, 0)


def write_mat(path: Path, variable_name: str, array: np.ndarray, compressed: bool) -> None:
    scipy.io.savemat(
        path,
        {variable_name: array},
        do_compression=compressed,
    )


def write_envi(hdr_path: Path, raw_path: Path, array_yxb: np.ndarray) -> None:
    dtype = np.dtype(array_yxb.dtype)
    if dtype not in ENVI_DTYPES:
        raise ValueError(f"Unsupported ENVI dtype: {dtype}")
    lines, samples, bands = array_yxb.shape
    header = "\n".join(
        [
            "ENVI",
            f"samples = {samples}",
            f"lines   = {lines}",
            f"bands   = {bands}",
            "header offset = 0",
            "file type = ENVI Standard",
            f"data type = {ENVI_DTYPES[dtype]}",
            "interleave = bip",
            "byte order = 0",
            "",
        ]
    )
    hdr_path.write_text(header)
    array_yxb.tofile(raw_path)


def write_omezarr(zarr_path: Path, cube_cyx: np.ndarray) -> None:
    if zarr_path.exists():
        if zarr_path.is_dir():
            for item in sorted(zarr_path.rglob("*"), reverse=True):
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    item.rmdir()
            zarr_path.rmdir()
        else:
            zarr_path.unlink()
    store = parse_url(str(zarr_path), mode="w").store
    root = zarr.group(store=store)
    write_image(
        image=cube_cyx,
        group=root,
        axes="cyx",
        scaler=Scaler(max_layer=4),
    )


def main() -> None:
    configure_runtime()
    args = parse_args()

    npy_path = Path(args.npy_file)
    output_stem = Path(args.output_stem) if args.output_stem else npy_path.with_suffix("")
    array = np.load(npy_path)
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D array, got shape {array.shape}")

    array_yxb = move_spectral_last(array, args.spectral_axis)
    cube_cyx = move_spectral_first(array, args.spectral_axis)

    compressed_mat = output_stem.with_suffix(".mat")
    uncompressed_mat = output_stem.with_name(f"{output_stem.name}_uncompressed").with_suffix(".mat")
    envi_hdr = output_stem.with_suffix(".hdr")
    envi_raw = output_stem.with_suffix(".raw")
    omezarr_path = output_stem.with_suffix(".ome.zarr")

    write_mat(compressed_mat, args.mat_var, array_yxb, compressed=True)
    write_mat(uncompressed_mat, args.mat_var, array_yxb, compressed=False)
    write_envi(envi_hdr, envi_raw, array_yxb)
    write_omezarr(omezarr_path, cube_cyx)

    print(f"Wrote compressed MAT: {compressed_mat}")
    print(f"Wrote uncompressed MAT: {uncompressed_mat}")
    print(f"Wrote ENVI header: {envi_hdr}")
    print(f"Wrote ENVI raw: {envi_raw}")
    print(f"Wrote OME-Zarr: {omezarr_path}")


if __name__ == "__main__":
    main()
