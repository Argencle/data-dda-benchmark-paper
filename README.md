# Data Availability - DDA Benchmark Paper

This repository contains the data, scripts, and figures used in the paper.

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
- `logs_MPI_laptop/`, `logs_MPI_cluster/`: CPU/MPI results.
- `logs_GPU_2000Ada/`, `logs_GPU_6000Ada/`, `logs_GPU_cluster/`: GPU results.
- `Figures/`: aggregated figure datasets and final figures.
- `run_MPI_random_and_extract.sh`: runs MPI/CPU simulations and extracts metrics.
- `run_GPU_random_and_extract.sh`: runs GPU simulations and extracts metrics.
- `sort_datafiles.py`: standard sorting of result CSV files.
- `fill_time_in_datafiles.py`: fills elapsed times (cluster/SLURM case).
- `make_figure1_data.py`, `make_figure2_data.py`, `make_figure3_data.py`, `make_appendix_data.py`: build figure CSV datasets.
- `plot_figure1.py`, `plot_figure2.py`, `plot_figure3.py`, `plot_appendix.py`: generate PDF figures in `Figures/`.

## Python Requirements

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install pandas matplotlib
```

## Reproducing Code Modifications

### 1) ADDA

Expected base version: commit `b03d6480b5f41b88abe0b201c847da780d1efe56`.

```bash
git clone https://github.com/adda-team/adda.git
cd adda
git checkout b03d6480b5f41b88abe0b201c847da780d1efe56
git apply --check ../patches/adda_b03d648.patch
git apply ../patches/adda_b03d648.patch
```

Then compile ADDA.

### 2) DDSCAT

Expected base version: `ddscat7.3.4_250505`.

```bash
patch --dry-run -p0 < patches/ddscat_250505.patch
patch -p0 < patches/ddscat_250505.patch
```

Then compile DDSCAT.

## Figure Reproduction Pipeline

If raw CSV files are already present:

```bash
python sort_datafiles.py
python fill_time_in_datafiles.py
python make_figure1_data.py
python make_figure2_data.py
python make_figure3_data.py
python make_appendix_data.py
python plot_figure1.py
python plot_figure2.py
python plot_figure3.py
python plot_appendix.py
```

Outputs are written to `Figures/`.

## Notes

- `run_MPI_random_and_extract.sh` and `run_GPU_random_and_extract.sh` contain local paths that must be adapted.
