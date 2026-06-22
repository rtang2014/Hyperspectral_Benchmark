import argparse
from pathlib import Path

import h5py


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rewrite a simple MATLAB v7.3 MAT file without HDF5 dataset compression."
    )
    parser.add_argument("src", help="Source .mat file")
    parser.add_argument("dst", help="Destination .mat file")
    return parser.parse_args()


def copy_attrs(src_obj, dst_obj) -> None:
    for key, value in src_obj.attrs.items():
        dst_obj.attrs[key] = value


def copy_item(src_obj, dst_parent, name: str) -> None:
    if isinstance(src_obj, h5py.Group):
        dst_group = dst_parent.create_group(name)
        copy_attrs(src_obj, dst_group)
        for child_name, child_obj in src_obj.items():
            copy_item(child_obj, dst_group, child_name)
        return

    dst_dataset = dst_parent.create_dataset(
        name,
        data=src_obj[()],
        dtype=src_obj.dtype,
    )
    copy_attrs(src_obj, dst_dataset)


def main() -> None:
    args = parse_args()
    src = Path(args.src)
    dst = Path(args.dst)
    header = src.read_bytes()[:512]

    with h5py.File(src, "r") as src_file, h5py.File(dst, "w", userblock_size=512) as dst_file:
        copy_attrs(src_file, dst_file)
        for name, obj in src_file.items():
            copy_item(obj, dst_file, name)

    with dst.open("r+b") as handle:
        handle.write(header)

    print(f"Wrote uncompressed MAT v7.3-style file to {dst}")


if __name__ == "__main__":
    main()
