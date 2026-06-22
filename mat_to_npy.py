import argparse
from pathlib import Path

import numpy as np
import scipy.io


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a MATLAB .mat file variable into a NumPy .npy file."
    )
    parser.add_argument("mat_file", help="Path to the input .mat file")
    parser.add_argument(
        "-v",
        "--variable",
        help="Variable name inside the MAT file. If omitted, the script auto-selects when possible.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output .npy path. Defaults to <mat_file_stem>_<variable>.npy",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List user variables in the MAT file and exit.",
    )
    return parser.parse_args()


def _visible_keys(mapping: dict) -> list[str]:
    return [key for key in mapping.keys() if not key.startswith("__")]


def _load_standard_mat(mat_path: Path) -> dict[str, np.ndarray]:
    return {key: value for key, value in scipy.io.loadmat(mat_path).items() if not key.startswith("__")}


def _load_hdf5_mat(mat_path: Path) -> dict[str, np.ndarray]:
    import h5py

    with h5py.File(mat_path, "r") as handle:
        result: dict[str, np.ndarray] = {}
        for key in handle.keys():
            obj = handle[key]
            if isinstance(obj, h5py.Dataset):
                result[key] = np.array(obj)
        return result


def load_mat_variables(mat_path: Path) -> dict[str, np.ndarray]:
    try:
        return _load_standard_mat(mat_path)
    except NotImplementedError as exc:
        if "matlab v7.3" not in str(exc).lower():
            raise
        return _load_hdf5_mat(mat_path)


def choose_variable(variables: dict[str, np.ndarray], requested: str | None) -> str:
    if requested is not None:
        if requested not in variables:
            available = ", ".join(sorted(variables))
            raise KeyError(f"Variable '{requested}' not found. Available: {available}")
        return requested
    if len(variables) == 1:
        return next(iter(variables))
    if "Image" in variables:
        return "Image"
    available = ", ".join(sorted(variables))
    raise ValueError(
        "Multiple variables found; specify one with --variable. "
        f"Available: {available}"
    )


def default_output_path(mat_path: Path, variable: str) -> Path:
    return mat_path.with_name(f"{mat_path.stem}_{variable}.npy")


def main() -> None:
    args = parse_args()
    mat_path = Path(args.mat_file)
    variables = load_mat_variables(mat_path)

    if args.list:
        for key in sorted(variables):
            value = variables[key]
            print(f"{key}\tshape={value.shape}\tdtype={value.dtype}")
        return

    variable = choose_variable(variables, args.variable)
    output_path = Path(args.output) if args.output else default_output_path(mat_path, variable)
    array = np.asarray(variables[variable])
    np.save(output_path, array)
    print(
        f"Saved '{variable}' from {mat_path} to {output_path} "
        f"with shape={array.shape} dtype={array.dtype}"
    )


if __name__ == "__main__":
    main()
