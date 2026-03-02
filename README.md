[![DOI](https://zenodo.org/badge/1106070872.svg)](https://doi.org/10.5281/zenodo.18777306)

# Data Availability - DDA Benchmark Paper

This repository contains the data, scripts, and figures used in the paper.

## Related Repository

The benchmark software used in this work is available at:
- https://github.com/Argencle/dda-bench.git

## License

- Source code is licensed under the MIT License.
- Data and figures are licensed under CC-BY-4.0.

## Repository Purpose

- Provide the raw benchmark results (logs + CSV) used in the paper.
- Provide post-processing scripts to rebuild figure datasets.
- Provide plotting scripts to regenerate the final PDF figures.
- Provide patches required to reproduce the modified third-party codes:
  - ADDA
  - DDSCAT

## Repository Structure

- `patches/`
- `patches/adda_b03d648.patch`: ADDA patch.
- `patches/ddscat_250505.patch`: DDSCAT patch.
- `datas/`
- `logs_MPI_laptop/`, `logs_MPI_cluster/`: CPU/MPI results.
- `logs_GPU_2000Ada/`, `logs_GPU_6000Ada/`, `logs_GPU_cluster/`: GPU results.
- In `logs*`, run logs are stored as `.zip` archives; only aggregated files are kept uncompressed.
- `*.csv`: aggregated figure datasets.
- `scripts/`
- `run_MPI_random_and_extract.sh`: runs MPI/CPU simulations and extracts metrics.
- `run_GPU_random_and_extract.sh`: runs GPU simulations and extracts metrics.
- `sort_datafiles.py`: standard sorting of result CSV files.
- `fill_time_in_datafiles.py`: fills elapsed times (cluster/SLURM case).
- `make_figure1_data.py`, `make_figure2_data.py`, `make_figure3_data.py`, `make_appendix_data.py`: build figure CSV datasets.
- `plot_figure1.py`, `plot_figure2.py`, `plot_figure3.py`, `plot_appendix.py`: generate PDF figures in `figures/`.
- `figures/`: output folder storing generated PDF figures.

## Python Requirements

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Reproducing Code Modifications

### 1) ADDA

Expected base version: commit [b03d6480b5f41b88abe0b201c847da780d1efe56](https://github.com/adda-team/adda/tree/b03d6480b5f41b88abe0b201c847da780d1efe56).

Optional (without `git clone` and `git checkout`): download the ZIP snapshot of that commit directly from
https://github.com/adda-team/adda/archive/b03d6480b5f41b88abe0b201c847da780d1efe56.zip,
extract it, and apply the same patch in the extracted source tree.

```bash
git clone https://github.com/adda-team/adda.git
cd adda
git checkout b03d6480b5f41b88abe0b201c847da780d1efe56
git apply --check ../patches/adda_b03d648.patch
git apply ../patches/adda_b03d648.patch
```

Then compile ADDA.

### 2) DDSCAT

Expected base version: [7.3.4_250505](http://ddscat.wikidot.com/downloads).

Download the source ZIP from the official downloads page and extract it first.
Then run the patch commands:

```bash
patch --dry-run -p0 < patches/ddscat_250505.patch
patch -p0 < patches/ddscat_250505.patch
```

Then compile DDSCAT.

## Regenerate Simulation Logs

All run logs are already provided in `logs*` as `.zip` archives.

If you want to rerun simulations and re-extract metrics:

```bash
bash scripts/run_MPI_random_and_extract.sh
bash scripts/run_GPU_random_and_extract.sh
```

**Note**: both scripts contain local paths that must be adapted.

## Update aggregated CSV files from logs

Aggregated CSV files are already included in `logs*` (uncompressed part), as `*_results_sorted.csv`.

Run:

```bash
python scripts/sort_datafiles.py
python scripts/fill_time_in_datafiles.py
```

**Note**: 
- the ‘logs*’ folders must be unzipped in order to use these scripts.
- `sort_datafiles.py` will not reproduce the exact repository versions of `ddscat_results_sorted.csv` and `ifdda_results_sorted.csv` unless 4 outlier runs are removed. In the repository, these 4 lines were manually removed because they correspond to broken runs with unrealistic values.

## Build final figure datasets

Final figure datasets are already provided in `datas/` as CSV files.

Run:

```bash
python scripts/make_figure1_data.py
python scripts/make_figure2_data.py
python scripts/make_figure3_data.py
python scripts/make_appendix_data.py
```

## Figure Reproduction (Plots Only)

Run:

```bash
python scripts/plot_figure1.py
python scripts/plot_figure2.py
python scripts/plot_figure3.py
python scripts/plot_appendix.py
```

Outputs are written to `figures/`.
