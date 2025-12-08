#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


def grouped_stacked_bars_with_overlay_ax(
    ax,
    title,
    x_labels,
    solver_series,
    total_series,
    overlay_total_series=None,  # e.g., {"IFDDA DP": [...], "IFDDA SP": [...]}
    overlay_label="10-core wall time",
    ylabel="Seconds",
    solver_alpha=0.95,
    overhead_alpha=0.35,
    hatch_pattern="//",
    seg_legend_loc=None,
    code_legend_loc=None,
    annotate_fontsize=9,
    overlay_offset=0.6,
):
    """
    Draw on a provided Axes `ax`:
      - grouped bars per code
      - bottom segment: solver time (1 core / baseline OMP)
      - top segment: overhead = total - solver
      - optional hatched overlay bar for e.g. 10-core TOTAL
      - annotate TOTAL and solver values, and overlay values above hatched bars
    """
    series_names = list(solver_series.keys())
    assert set(series_names) == set(
        total_series.keys()
    ), "Solver/Total keys must match."

    n_series = len(series_names)
    n_groups = len(x_labels)

    for s in series_names:
        assert len(solver_series[s]) == n_groups
        assert len(total_series[s]) == n_groups
    if overlay_total_series:
        for s, arr in overlay_total_series.items():
            assert (
                s in series_names
            ), f"Overlay key {s} must match a base series key"
            assert (
                len(arr) == n_groups
            ), f"Overlay series length mismatch for {s}"

    x = np.arange(n_groups)
    width = 0.8 / max(1, n_series)

    code_patches = []
    bar_positions = {}

    # Draw stacked bars for each code
    for i, s in enumerate(series_names):
        sol = np.array(solver_series[s], dtype=float)
        tot = np.array(total_series[s], dtype=float)

        # Replace NaNs by zero for plotting, but keep original values for labels
        sol_plot = np.nan_to_num(sol, nan=0.0)
        tot_plot = np.nan_to_num(tot, nan=0.0)
        over = np.maximum(tot_plot - sol_plot, 0.0)

        x_off = x + i * width - 0.4 + (width * n_series) / 2
        bar_positions[s] = x_off

        b1 = ax.bar(x_off, sol_plot, width, alpha=solver_alpha)
        face = b1.patches[0].get_facecolor()

        ax.bar(
            x_off,
            over,
            width,
            bottom=sol_plot,
            color=face,
            alpha=overhead_alpha,
        )

        # Annotate totals and solver (only if finite)
        for xi, si, oi, ti in zip(x_off, sol_plot, over, tot):
            if np.isfinite(ti) and ti > 0:
                ax.text(
                    xi,
                    si + oi,
                    f"{ti:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=annotate_fontsize,
                )
        for xi, si, ti in zip(x_off, sol_plot, sol):
            if np.isfinite(ti) and si > 0:
                ax.text(
                    xi,
                    si / 2.0,
                    f"{ti:.1f}",
                    ha="center",
                    va="center",
                    fontsize=annotate_fontsize,
                )

        code_patches.append(
            Patch(facecolor=face, edgecolor=face, alpha=solver_alpha, label=s)
        )

    # Hatched overlays (e.g., 10-core totals) + annotate above hatch
    overlay_patch = None
    if overlay_total_series:
        for s, arr in overlay_total_series.items():
            x_off = bar_positions[s]
            arr = np.array(arr, dtype=float)
            arr_plot = np.nan_to_num(arr, nan=0.0)

            ax.bar(
                x_off,
                arr_plot,
                width * 0.95,
                fill=False,
                hatch=hatch_pattern,
                linewidth=0.6,
                label=overlay_label if overlay_patch is None else None,
            )
            # Values above the hatched bars
            for xi, yi, val in zip(x_off, arr_plot, arr):
                if np.isfinite(val) and yi > 0:
                    ax.text(
                        xi,
                        yi + overlay_offset,
                        f"{val:.1f}",
                        ha="center",
                        va="bottom",
                        fontsize=annotate_fontsize,
                    )
            if overlay_patch is None:
                overlay_patch = Patch(
                    fill=False, hatch=hatch_pattern, label=overlay_label
                )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, pad=6)

    # Legends (optional)
    if seg_legend_loc:
        seg_patches = [
            Patch(alpha=solver_alpha),
            Patch(alpha=overhead_alpha),
        ]
        leg1 = ax.legend(
            seg_patches,
            ["Solver", "Wall time"],
            loc=seg_legend_loc,
            ncols=1,
            frameon=False,
        )
        ax.add_artist(leg1)

    if code_legend_loc:
        patches = code_patches.copy()
        if overlay_patch:
            patches.append(overlay_patch)
        ax.legend(
            patches,
            [p.get_label() for p in patches],
            loc=code_legend_loc,
            ncols=1,
            frameon=False,
        )


# -------------------------------------------------------------------
# Helpers to load & label data
# -------------------------------------------------------------------


def load_ifdda_gpu_data():
    """
    Load all IFDDA GPU CSVs, add a 'gpu_label' column, and concatenate.

    Files:
      - logs_GPU_cluster/ifdda_gpu_results_sorted.csv   (cluster: A100 + H200, with 'partition')
      - logs_GPU_6000Ada/ifdda_gpu_results.csv          (RTX 6000 Ada, no partition)
      - logs_GPU_2000Ada/ifdda_gpu_results_sorted.csv   (RTX 2000 Ada, no partition)
    """
    sources = [
        {
            "path": "logs_GPU_cluster/ifdda_gpu_results_sorted.csv",
            "type": "cluster",
        },
        {
            "path": "logs_GPU_6000Ada/ifdda_gpu_results.csv",
            "type": "single",
            "gpu_label": "RTX 6000 Ada",
        },
        {
            "path": "logs_GPU_2000Ada/ifdda_gpu_results_sorted.csv",
            "type": "single",
            "gpu_label": "RTX 2000 Ada",
        },
    ]

    dfs = []

    # Map 'partition' -> GPU label for the cluster file
    cluster_partition_to_label = {
        "gpu_all": "NVIDIA A100",
        "gpu_h200": "NVIDIA H200",
    }

    for src in sources:
        df = pd.read_csv(src["path"])
        df.columns = [c.lower() for c in df.columns]

        if src["type"] == "cluster":
            if "partition" not in df.columns:
                raise ValueError(
                    "Cluster IFDDA file is expected to have a 'partition' column."
                )
            df["gpu_label"] = df["partition"].map(cluster_partition_to_label)
        else:
            # Single-GPU files: assign constant label
            df["gpu_label"] = src["gpu_label"]

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def load_adda_gpu_data():
    """
    Load all ADDA GPU CSVs, add a 'gpu_label' column, and concatenate.

    Files:
      - logs_GPU_cluster/adda_gpu_results_sorted.csv
      - logs_GPU_6000Ada/adda_gpu_results.csv
      - logs_GPU_2000Ada/adda_gpu_results_sorted.csv
    """
    sources = [
        {
            "path": "logs_GPU_cluster/adda_gpu_results_sorted.csv",
            "type": "cluster",
        },
        {
            "path": "logs_GPU_6000Ada/adda_gpu_results.csv",
            "type": "single",
            "gpu_label": "RTX 6000 Ada",
        },
        {
            "path": "logs_GPU_2000Ada/adda_gpu_results_sorted.csv",
            "type": "single",
            "gpu_label": "RTX 2000 Ada",
        },
    ]

    dfs = []

    cluster_partition_to_label = {
        "gpu_all": "NVIDIA A100",
        "gpu_h200": "NVIDIA H200",
    }

    for src in sources:
        df = pd.read_csv(src["path"])
        df.columns = [c.lower() for c in df.columns]

        if src["type"] == "cluster":
            if "partition" not in df.columns:
                raise ValueError(
                    "Cluster ADDA file is expected to have a 'partition' column."
                )
            df["gpu_label"] = df["partition"].map(cluster_partition_to_label)
        else:
            df["gpu_label"] = src["gpu_label"]

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def get_ifdda_times(df_ifdda, N, gpu_label, precision, omp):
    """
    Extract mean solver_time and total_time for IFDDA GPU.

    Parameters
    ----------
    df_ifdda : DataFrame (all GPUs, all Ns)
    N        : int  (grid size)
    gpu_label: str  (values in column 'gpu_label')
    precision: 'dp' (double) or 'sp' (single)
    omp      : int  (e.g. 1 or 10)

    Returns
    -------
    solver_mean, total_mean  (floats or np.nan if no data)
    """
    # Executable names according to precision (both variants supported)
    if precision.lower() == "dp":
        exe_names = ["ifdda_gpu", "ifdda_GPU"]
    else:
        exe_names = ["ifdda_gpu_sp", "ifdda_GPU_single"]

    # Columns are lowercase: n, omp, exe, gpu_label, total_time, solver_time
    mask = (
        (df_ifdda["n"] == N)
        & (df_ifdda["gpu_label"] == gpu_label)
        & (df_ifdda["exe"].isin(exe_names))
        & (df_ifdda["omp"] == omp)
    )

    sub = df_ifdda.loc[mask]

    if sub.empty:
        return np.nan, np.nan

    # Use the mean over repetitions
    if "solver_time" not in sub.columns or "total_time" not in sub.columns:
        raise ValueError(
            "IFDDA CSV must contain 'solver_time' and 'total_time' columns."
        )

    solver_mean = sub["solver_time"].mean()
    total_mean = sub["total_time"].mean()

    return solver_mean, total_mean


def get_adda_times(df_adda, N, gpu_label):
    """
    Extract mean solver_time and total_wall_time (or equivalent) for ADDA GPU.

    Parameters
    ----------
    df_adda  : DataFrame
    N        : int
    gpu_label: str (value in column 'gpu_label')

    Returns
    -------
    solver_mean, total_mean
    """
    # Executable name for ADDA GPU
    exe_names = ["adda_ocl", "adda_OCL"]  # be a bit permissive

    mask = (
        (df_adda["n"] == N)
        & (df_adda["gpu_label"] == gpu_label)
        & (df_adda["exe"].isin(exe_names))
    )

    sub = df_adda.loc[mask]

    if sub.empty:
        return np.nan, np.nan

    if "solver_time" not in sub.columns:
        raise ValueError("ADDA CSV must contain 'solver_time' column.")

    # Depending on your CSV, total time may be 'total_wall_time', 'total_time' or 'elapsed_seconds'
    if "total_wall_time" in sub.columns:
        total_col = "total_wall_time"
    elif "total_time" in sub.columns:
        total_col = "total_time"
    elif "elapsed_seconds" in sub.columns:
        total_col = "elapsed_seconds"
    else:
        raise ValueError(
            "ADDA CSV must contain 'total_wall_time', 'total_time' or 'elapsed_seconds'."
        )

    solver_mean = sub["solver_time"].mean()
    total_mean = sub[total_col].mean()

    return solver_mean, total_mean


def main():
    # ------------------------------------------------------------------
    # 1) Labels on x-axis for N=150 and N=250
    # ------------------------------------------------------------------
    x_labels_150 = [
        "NVIDIA A100",
        "NVIDIA H200",
        "RTX 6000 Ada",
        "RTX 2000 Ada",
    ]
    x_labels_250 = [
        "NVIDIA A100",
        "NVIDIA H200",
        "RTX 6000 Ada",
        # RTX 2000 Ada can be added here if you have N=250 data for this GPU
    ]

    # OMP used for "baseline" (full filled bar) and "overlay" (hatched)
    # If your current data uses OMP=2 and OMP=10, change baseline_omp=2.
    baseline_omp = 1
    overlay_omp = 10

    # ------------------------------------------------------------------
    # 2) Load data from CSVs
    # ------------------------------------------------------------------
    df_ifdda = load_ifdda_gpu_data()
    df_adda = load_adda_gpu_data()

    # ------------------------------------------------------------------
    # 3) Build data dictionaries from CSV
    # ------------------------------------------------------------------

    # N = 150
    solver_150 = {
        "ADDA": [],
        "IFDDA DP": [],
        "IFDDA SP": [],
    }
    total_150 = {
        "ADDA": [],
        "IFDDA DP": [],
        "IFDDA SP": [],
    }
    overlay_150 = {
        "IFDDA DP": [],
        "IFDDA SP": [],
    }

    for gpu_label in x_labels_150:
        # ADDA (baseline)
        adda_solver, adda_total = get_adda_times(
            df_adda, N=150, gpu_label=gpu_label
        )
        solver_150["ADDA"].append(adda_solver)
        total_150["ADDA"].append(adda_total)

        # IFDDA DP
        ifdda_dp_solver_1, ifdda_dp_total_1 = get_ifdda_times(
            df_ifdda,
            N=150,
            gpu_label=gpu_label,
            precision="dp",
            omp=baseline_omp,
        )
        _, ifdda_dp_total_ov = get_ifdda_times(
            df_ifdda,
            N=150,
            gpu_label=gpu_label,
            precision="dp",
            omp=overlay_omp,
        )
        solver_150["IFDDA DP"].append(ifdda_dp_solver_1)
        total_150["IFDDA DP"].append(ifdda_dp_total_1)
        overlay_150["IFDDA DP"].append(ifdda_dp_total_ov)

        # IFDDA SP
        ifdda_sp_solver_1, ifdda_sp_total_1 = get_ifdda_times(
            df_ifdda,
            N=150,
            gpu_label=gpu_label,
            precision="sp",
            omp=baseline_omp,
        )
        _, ifdda_sp_total_ov = get_ifdda_times(
            df_ifdda,
            N=150,
            gpu_label=gpu_label,
            precision="sp",
            omp=overlay_omp,
        )
        solver_150["IFDDA SP"].append(ifdda_sp_solver_1)
        total_150["IFDDA SP"].append(ifdda_sp_total_1)
        overlay_150["IFDDA SP"].append(ifdda_sp_total_ov)

    # N = 250
    solver_250 = {
        "ADDA": [],
        "IFDDA DP": [],
        "IFDDA SP": [],
    }
    total_250 = {
        "ADDA": [],
        "IFDDA DP": [],
        "IFDDA SP": [],
    }
    overlay_250 = {
        "IFDDA DP": [],
        "IFDDA SP": [],
    }

    for gpu_label in x_labels_250:
        # ADDA
        adda_solver, adda_total = get_adda_times(
            df_adda, N=250, gpu_label=gpu_label
        )
        solver_250["ADDA"].append(adda_solver)
        total_250["ADDA"].append(adda_total)

        # IFDDA DP
        ifdda_dp_solver_1, ifdda_dp_total_1 = get_ifdda_times(
            df_ifdda,
            N=250,
            gpu_label=gpu_label,
            precision="dp",
            omp=baseline_omp,
        )
        _, ifdda_dp_total_ov = get_ifdda_times(
            df_ifdda,
            N=250,
            gpu_label=gpu_label,
            precision="dp",
            omp=overlay_omp,
        )
        solver_250["IFDDA DP"].append(ifdda_dp_solver_1)
        total_250["IFDDA DP"].append(ifdda_dp_total_1)
        overlay_250["IFDDA DP"].append(ifdda_dp_total_ov)

        # IFDDA SP
        ifdda_sp_solver_1, ifdda_sp_total_1 = get_ifdda_times(
            df_ifdda,
            N=250,
            gpu_label=gpu_label,
            precision="sp",
            omp=baseline_omp,
        )
        _, ifdda_sp_total_ov = get_ifdda_times(
            df_ifdda,
            N=250,
            gpu_label=gpu_label,
            precision="sp",
            omp=overlay_omp,
        )
        solver_250["IFDDA SP"].append(ifdda_sp_solver_1)
        total_250["IFDDA SP"].append(ifdda_sp_total_1)
        overlay_250["IFDDA SP"].append(ifdda_sp_total_ov)

    # ------------------------------------------------------------------
    # 4) Plot
    # ------------------------------------------------------------------
    plt.rcParams.update(
        {
            "figure.figsize": (9.5, 9.5),
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(9.5, 9.5), sharex=False
    )
    fig.subplots_adjust(hspace=0.35)

    # (a) N=150
    grouped_stacked_bars_with_overlay_ax(
        ax=ax_top,
        title=r"",
        x_labels=x_labels_150,
        solver_series=solver_150,
        total_series=total_150,
        overlay_total_series=overlay_150,
        ylabel="Seconds",
        seg_legend_loc="upper center",
        code_legend_loc="upper left",
        annotate_fontsize=9,
        overlay_offset=0.6,
    )

    # (b) N=250
    grouped_stacked_bars_with_overlay_ax(
        ax=ax_bot,
        title=r"",
        x_labels=x_labels_250,
        solver_series=solver_250,
        total_series=total_250,
        overlay_total_series=overlay_250,
        ylabel="Seconds",
        seg_legend_loc=None,
        code_legend_loc=None,
        annotate_fontsize=9,
        overlay_offset=1.0,
    )

    plt.tight_layout()
    plt.savefig("Figure3_from_csv.pdf", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
