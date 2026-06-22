# Manuscript Code

Python scripts used for the hyperspectral storage-format benchmark manuscript.

## Core benchmark scripts

- `convert_hsi.py`: shared readers, OME-Zarr conversion, benchmark operations, PCA, and optional napari viewing.
- `benchmark_canola_variants.py`: Canola format benchmark.
- `benchmark_beef_variants.py`: Beef format benchmark.
- `benchmark_geo_variants.py`: Geo format benchmark.

## Figure scripts

- `generate_paper_figures.py`: workflow, file-size, time-vs-size, task-runtime, and supplementary figures.
- `generate_visual_benchmark_comparison.py`: combined visual benchmark comparison figure.
- `generate_napari_benchmark_views.py`: napari-style benchmark views, including the Beef example.
- `generate_visual_figures.py`: dataset overview, spectra, and PCA visual figures.
- `plot_size_trend.py`: standalone time-vs-size trend figure.

## Data preparation utilities

- `mat_to_npy.py`: convert MATLAB data to NumPy.
- `npy_to_hyperspectral_formats.py`: create additional hyperspectral format variants.
- `mat73_uncompress.py`: create uncompressed MATLAB development controls.

## Notes

The scripts expect the benchmark input data, CSV outputs, and report files to be available in the working data directory. If running these scripts from this folder while the data remain in `../tests`, pass `--base-dir ../tests` for figure-generation scripts that support it.
