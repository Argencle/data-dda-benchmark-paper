import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb


def add_legend_table(ax, code_colors, solver_alpha=0.95, overhead_alpha=0.35):
    """
    Tableau de légende :

    Lignes   : ADDA, IFDDA DP, IFDDA SP
    Colonnes : FFTs, Solver, Wall, 10-core
    """
    codes = ["ADDA", "IFDDA DP", "IFDDA SP"]
    col_labels = ["FFTs", "Solver", "Wall", "10-core"]

    n_rows = len(codes)
    n_cols = len(col_labels)

    # Texte vide, on n’affiche que la couleur
    cell_text = [["" for _ in range(n_cols)] for _ in range(n_rows)]

    def rgba(c, a=1.0):
        r, g, b = to_rgb(c)
        return (r, g, b, a)

    # Couleurs de fond des cellules
    cell_colours = [
        [(0.0, 0.0, 0.0, 0.0) for _ in range(n_cols)] for _ in range(n_rows)
    ]

    for i, code in enumerate(codes):
        base = code_colors[code]
        solver_col = lighten_color(base, 0.6)

        # FFTs seulement pour ADDA (colonne 0)
        if code == "ADDA":
            cell_colours[i][0] = rgba(base, 1.0)

        # Solver (colonne 1)
        cell_colours[i][1] = rgba(solver_col, solver_alpha)

        # Wall / overhead (colonne 2)
        cell_colours[i][2] = rgba(solver_col, overhead_alpha)
        # colonne 3 (10-core) reste transparente, on dessinera un trait

    table = ax.table(
        cellText=cell_text,
        cellColours=cell_colours,
        rowLabels=codes,
        colLabels=col_labels,
        cellLoc="center",
        rowLoc="center",
        loc="upper left",
        bbox=[0.08, 0.78, 0.20, 0.20],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)

    for key, cell in table.get_celld().items():
        cell.set_linewidth(0.3)

    # ------------------------------
    # Traits "10-core" dans la DERNIÈRE
    # colonne de données (index = n_cols - 1)
    # ------------------------------
    last_data_col = n_cols - 1  # 0..n_cols-1 sont les colonnes de données

    for i, code in enumerate(codes):
        if not code.startswith("IFDDA"):
            continue

        cell = table[(i + 1, last_data_col)]
        x, y = cell.get_xy()
        w, h = cell.get_width(), cell.get_height()

        line_DP = Line2D(
            [x + 0.95 * w, x + 1.09 * w],
            [y + 15.95 * h, y + 15.95 * h],
            transform=ax.transAxes,
            color="black",
            linewidth=3.0,
            clip_on=False,
            zorder=6,
        )
        line_SP = Line2D(
            [x + 0.95 * w, x + 1.09 * w],
            [y + 15.0 * h, y + 15.0 * h],
            transform=ax.transAxes,
            color="black",
            linewidth=3.0,
            clip_on=False,
            zorder=6,
        )
        ax.add_line(line_DP)
        ax.add_line(line_SP)


# ------------------------------------------------------------
# Color helpers
# ------------------------------------------------------------
def lighten_color(color, factor=0.7):
    """
    Return a lighter shade of the given matplotlib color by mixing with white.

    factor in (0,1): closer to 1 → closer to original color,
                     closer to 0 → closer to white.
    """
    r, g, b = to_rgb(color)
    return (
        (1 - factor) + factor * r,
        (1 - factor) + factor * g,
        (1 - factor) + factor * b,
    )


# ------------------------------------------------------------
# Grouped stacked bars with overlay
# ------------------------------------------------------------
def grouped_stacked_bars_with_overlay_ax(
    ax,
    title,
    x_labels,
    solver_series,
    total_series,
    overlay_total_series=None,  # e.g., {"IFDDA DP": [...], "IFDDA SP": [...]}
    matvec_series=None,  # e.g., {"ADDA": [...]}
    code_colors=None,  # dict: series_name -> base color
    xtick_series="IFDDA SP",  # which series to use to place x-ticks
    ylabel="Seconds",
    solver_alpha=0.95,
    overhead_alpha=0.35,
    annotate_fontsize=9,
    overlay_offset=0.6,
):
    """
    Draw on a provided Axes `ax`:
      - grouped bars per code (ADDA, IFDDA DP, IFDDA SP, ...)
      - if no matvec for a code:
          bottom segment: solver time (baseline OMP)
          top segment   : overhead = total - solver
      - if matvec for a code (e.g. ADDA):
          bottom : matvec time
          middle : solver - matvec
          top    : overhead = total - solver
      - overlay: thick black horizontal line representing (e.g.) 10-core TOTAL
      - annotate TOTAL, solver, and overlay values.
      - x tick labels aligned on the chosen `xtick_series`.
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
        if matvec_series is not None and s in matvec_series:
            assert len(matvec_series[s]) == n_groups

    if overlay_total_series:
        for s, arr in overlay_total_series.items():
            assert (
                s in series_names
            ), f"Overlay key {s} must match a base series key"
            assert (
                len(arr) == n_groups
            ), f"Overlay series length mismatch for {s}"

    # Choose colors for each series
    if code_colors is None:
        default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        code_colors = {
            name: default_colors[i % len(default_colors)]
            for i, name in enumerate(series_names)
        }

    x = np.arange(n_groups)
    width = 0.8 / max(1, n_series)

    bar_positions = {}  # store x-offsets per series

    # Draw stacked bars for each code
    for i, s in enumerate(series_names):
        base_color = code_colors[s]

        sol = np.array(solver_series[s], dtype=float)
        tot = np.array(total_series[s], dtype=float)

        # Replace NaNs for plotting (we still use original values for labels)
        sol_plot = np.nan_to_num(sol, nan=0.0)
        tot_plot = np.nan_to_num(tot, nan=0.0)

        over = np.maximum(tot_plot - sol_plot, 0.0)

        has_matvec = matvec_series is not None and s in matvec_series
        if has_matvec:
            mv = np.array(matvec_series[s], dtype=float)
            mv_plot = np.nan_to_num(mv, nan=0.0)
            # Safety: ensure matvec <= solver for display
            mv_plot = np.minimum(mv_plot, sol_plot)
            sol_rest_plot = np.maximum(sol_plot - mv_plot, 0.0)
        else:
            mv = None
            mv_plot = None
            sol_rest_plot = sol_plot.copy()

        x_off = x + i * width - 0.4 + (width * n_series) / 2
        bar_positions[s] = x_off

        solver_color = lighten_color(base_color, 0.6)

        # --- Draw bars ---
        if has_matvec and s == "ADDA":
            # For ADDA: darker matvec bar + lighter solver bar
            matvec_color = base_color

            # Matvec chunk (bottom)
            ax.bar(
                x_off,
                mv_plot,
                width,
                color=matvec_color,
                alpha=1.0,
            )

            # Remaining solver time above matvec
            ax.bar(
                x_off,
                sol_rest_plot,
                width,
                bottom=mv_plot,
                color=solver_color,
                alpha=solver_alpha,
            )

        else:
            # Other codes: the whole solver chunk in base color
            ax.bar(
                x_off,
                sol_plot,
                width,
                color=solver_color,
                alpha=solver_alpha,
            )

        # Overhead chunk (same hue, more transparent)
        ax.bar(
            x_off,
            over,
            width,
            bottom=sol_plot,
            color=solver_color,
            alpha=overhead_alpha,
        )

        # --- Annotations ---
        # Total time on top of stacked bar
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

        # Solver time label
        for xi, si, ti in zip(x_off, sol_plot, sol):
            if np.isfinite(ti) and si > 0:
                # For ADDA, put solver label slightly above the solver segment
                if s == "ADDA":
                    ax.text(
                        xi,
                        si + 0.5,
                        f"{ti:.1f}",
                        ha="center",
                        va="bottom",
                        fontsize=annotate_fontsize,
                    )
                else:
                    ax.text(
                        xi,
                        si / 2.0,
                        f"{ti:.1f}",
                        ha="center",
                        va="center",
                        fontsize=annotate_fontsize,
                    )

        # Matvec label (only if available)
        if has_matvec and mv is not None:
            for xi, mvi in zip(x_off, mv_plot):
                if mvi > 0:
                    ax.text(
                        xi,
                        mvi / 2.0,
                        f"{mvi:.1f}",
                        ha="center",
                        va="center",
                        fontsize=annotate_fontsize,
                        color="black",
                    )

    # --- Overlay: 10-core total as thick black horizontal line ---
    if overlay_total_series:
        for s, arr in overlay_total_series.items():
            x_off = bar_positions[s]
            arr = np.array(arr, dtype=float)
            arr_plot = np.nan_to_num(arr, nan=0.0)

            for xi, yi, val in zip(x_off, arr_plot, arr):
                if np.isfinite(val) and yi > 0:
                    # Thick black line centered on bar
                    ax.hlines(
                        yi,
                        xi - width * 0.45,
                        xi + width * 0.45,
                        linewidth=3.0,
                        color="black",
                    )
                    # Label above the overlay line
                    ax.text(
                        xi,
                        yi + overlay_offset,
                        f"{val:.1f}",
                        ha="center",
                        va="bottom",
                        fontsize=annotate_fontsize,
                    )
    oom_gpu_label = "RTX 2000 Ada"
    oom_series = ("IFDDA DP", "IFDDA SP")

    # ------------------------------------------------------------
    # OUT OF MEMORY markers (empty bars + text) for missing IFDDA data on RTX 2000 Ada
    # ------------------------------------------------------------
    if title == r"n_x=250":
        if oom_gpu_label in x_labels:
            oom_idx = x_labels.index(oom_gpu_label)

            # hauteur "raisonnable" pour le rectangle : basée sur le max total visible
            max_total = 0.0
            for s in series_names:
                arr = np.array(total_series[s], dtype=float)
                arr = np.nan_to_num(arr, nan=0.0)
                if arr.size > 0:
                    max_total = max(max_total, float(np.max(arr)))

            # fallback si tout est 0
            if max_total <= 0:
                max_total = 1.0

            oom_h = (
                0.9 * max_total
            )  # rectangle pas trop haut (ajuste si tu veux)
            oom_y = 0.0

            for s in oom_series:
                if s not in bar_positions:
                    continue

                xi = float(bar_positions[s][oom_idx])

                # rectangle vide (barre "fantôme")
                rect = plt.Rectangle(
                    (xi - width * 0.5, oom_y),
                    width,
                    oom_h,
                    fill=False,
                    linewidth=1.2,
                    edgecolor="black",
                    zorder=7,
                )
                ax.add_patch(rect)

                # texte au centre
                ax.text(
                    xi,
                    oom_y + 0.5 * oom_h,
                    "OUT OF\nMEMORY",
                    ha="center",
                    va="center",
                    fontsize=8,
                    rotation=0,
                    zorder=8,
                )

    # --- X ticks: align labels with the chosen series (e.g. "IFDDA SP") ---
    if xtick_series is not None and xtick_series in bar_positions:
        xtick_positions = bar_positions[xtick_series]
    else:
        # fallback: group centers
        xtick_positions = x

    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(x_labels, rotation=15, ha="right")
    ax.set_ylabel(ylabel)


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
    if precision.lower() == "dp":
        exe_names = [
            "ifdda_gpu",
        ]
    else:
        exe_names = [
            "ifdda_gpu_sp",
        ]

    mask = (
        (df_ifdda["n"] == N)
        & (df_ifdda["gpu_label"] == gpu_label)
        & (df_ifdda["exe"].isin(exe_names))
        & (df_ifdda["omp"] == omp)
    )

    sub = df_ifdda.loc[mask]

    if sub.empty:
        return np.nan, np.nan

    if "solver_time" not in sub.columns or "total_time" not in sub.columns:
        raise ValueError(
            "IFDDA CSV must contain 'solver_time' and 'total_time' columns."
        )

    solver_mean = sub["solver_time"].mean()
    total_mean = sub["total_time"].mean()

    solver_std = sub["solver_time"].std(ddof=1)
    total_std = sub["total_time"].std(ddof=1)

    # coefficient de variation (CV) en %
    solver_cv_pct = (
        100 * solver_std / solver_mean if solver_mean != 0 else np.nan
    )
    total_cv_pct = 100 * total_std / total_mean if total_mean != 0 else np.nan

    print(
        f"Mean IFDDA solver time: {solver_mean:.1f} with std: {solver_cv_pct:.2f}"
    )
    print(
        f"Mean IFDDA total time: {total_mean:.1f} with std: {total_cv_pct:.2f}"
    )
    with open("Figure3_mean_std_ifdda.txt", "a") as f:
        f.write(
            f"{N},{gpu_label},{precision},{omp},{solver_mean:.1f},{solver_cv_pct:.2f},{total_mean:.1f},{total_cv_pct:.2f}\n"
        )
    if solver_cv_pct > 5.0 or total_cv_pct > 5.0:
        print(sub)
    return solver_mean, total_mean


def get_adda_times(df_adda, N, gpu_label):
    """
    Extract mean solver_time, total time, and matvec/FFT time for ADDA GPU.

    Parameters
    ----------
    df_adda  : DataFrame
    N        : int
    gpu_label: str (value in column 'gpu_label')

    Returns
    -------
    solver_mean, total_mean, matvec_mean
    """
    exe_names = ["adda_ocl", "adda_OCL"]

    mask = (
        (df_adda["n"] == N)
        & (df_adda["gpu_label"] == gpu_label)
        & (df_adda["exe"].isin(exe_names))
        & (df_adda["solver"] == "bicgstab")
    )

    sub = df_adda.loc[mask]

    if sub.empty:
        return np.nan, np.nan, np.nan

    if "solver_time" not in sub.columns:
        raise ValueError("ADDA CSV must contain 'solver_time' column.")

    # Determine total time column
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

    # Matvec/FFT time (if available)
    if "matvec_time" in sub.columns:
        matvec_col = "matvec_time"
    elif "fft_time" in sub.columns:
        matvec_col = "fft_time"
    else:
        matvec_col = None

    solver_mean = sub["solver_time"].mean()
    total_mean = sub[total_col].mean()
    if matvec_col is not None:
        matvec_mean = sub[matvec_col].mean()
        matvec_std = sub[matvec_col].std(ddof=1)
    else:
        matvec_mean = np.nan
        matvec_std = np.nan

    solver_std = sub["solver_time"].std(ddof=1)
    total_std = sub[total_col].std(ddof=1)

    # coefficient de variation (CV) en %
    matvec_cv_pct = (
        100 * matvec_std / matvec_mean if matvec_mean != 0 else np.nan
    )
    solver_cv_pct = (
        100 * solver_std / solver_mean if solver_mean != 0 else np.nan
    )
    total_cv_pct = 100 * total_std / total_mean if total_mean != 0 else np.nan

    print(
        f"Mean ADDA matvec time: {matvec_mean:.1f} with std: {matvec_cv_pct:.2f}"
    )
    print(
        f"Mean ADDA solver time: {solver_mean:.1f} with std: {solver_cv_pct:.2f}"
    )
    print(
        f"Mean ADDA total time: {total_mean:.1f} with std: {total_cv_pct:.2f}"
    )
    with open("Figure3_mean_std_adda.txt", "a") as f:
        f.write(
            f"{N},{gpu_label},{matvec_mean:.1f},{matvec_cv_pct:.2f},{solver_mean:.1f},{solver_cv_pct:.2f},{total_mean:.1f},{total_cv_pct:.2f}\n"
        )
    if (solver_cv_pct > 5.0) or (total_cv_pct > 5.0):
        print(sub)

    return solver_mean, total_mean, matvec_mean


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    # X labels for each N
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
        "RTX 2000 Ada",
    ]

    with open("Figure3_mean_std_ifdda.txt", "a") as f:
        f.write(
            "N,GPU,Precision,OMP,SolverMean,SolverCVpct,TotalMean,TotalCVpct\n"
        )
    with open("Figure3_mean_std_adda.txt", "a") as f:
        f.write(
            "N,GPU,MatvecMean,MatvecCVpct,SolverMean,SolverCVpct,TotalMean,TotalCVpct\n"
        )

    baseline_omp = 1  # baseline (bar filled)
    overlay_omp = 10  # 10-core overlay line

    df_ifdda = load_ifdda_gpu_data()
    df_adda = load_adda_gpu_data()

    # ----------------- N = 150 -----------------
    solver_150 = {"ADDA": [], "IFDDA DP": [], "IFDDA SP": []}
    total_150 = {"ADDA": [], "IFDDA DP": [], "IFDDA SP": []}
    overlay_150 = {"IFDDA DP": [], "IFDDA SP": []}
    matvec_150 = {"ADDA": []}

    for gpu_label in x_labels_150:
        # ADDA
        adda_solver, adda_total, adda_matvec = get_adda_times(
            df_adda, N=150, gpu_label=gpu_label
        )
        solver_150["ADDA"].append(adda_solver)
        total_150["ADDA"].append(adda_total)
        matvec_150["ADDA"].append(adda_matvec)

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

    # ----------------- N = 250 -----------------
    solver_250 = {"ADDA": [], "IFDDA DP": [], "IFDDA SP": []}
    total_250 = {"ADDA": [], "IFDDA DP": [], "IFDDA SP": []}
    overlay_250 = {"IFDDA DP": [], "IFDDA SP": []}
    matvec_250 = {"ADDA": []}

    for gpu_label in x_labels_250:
        # ADDA
        adda_solver, adda_total, adda_matvec = get_adda_times(
            df_adda, N=250, gpu_label=gpu_label
        )
        solver_250["ADDA"].append(adda_solver)
        total_250["ADDA"].append(adda_total)
        matvec_250["ADDA"].append(adda_matvec)

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

    # ----------------- Plotting -----------------
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

    solver_alpha = 0.95
    overhead_alpha = 0.35

    # Define base colors per code
    code_colors = {
        "ADDA": "#1f77b4",  # blue
        "IFDDA DP": "#ff7f0e",  # orange
        "IFDDA SP": "#2ca02c",  # green
    }

    # (a) N=150
    grouped_stacked_bars_with_overlay_ax(
        ax=ax_top,
        title=r"",
        x_labels=x_labels_150,
        solver_series=solver_150,
        total_series=total_150,
        overlay_total_series=overlay_150,
        matvec_series=matvec_150,
        code_colors=code_colors,
        xtick_series="IFDDA SP",  # align x-ticks on IFDDA SP
        ylabel="Seconds",
        annotate_fontsize=9,
        overlay_offset=0.6,
        solver_alpha=solver_alpha,
        overhead_alpha=overhead_alpha,
    )

    add_legend_table(
        ax_top,
        code_colors,
        solver_alpha=solver_alpha,
        overhead_alpha=overhead_alpha,
    )

    # (b) N=250
    grouped_stacked_bars_with_overlay_ax(
        ax=ax_bot,
        title=r"n_x=250",
        x_labels=x_labels_250,
        solver_series=solver_250,
        total_series=total_250,
        overlay_total_series=overlay_250,
        matvec_series=matvec_250,
        code_colors=code_colors,
        xtick_series="IFDDA SP",
        ylabel="Seconds",
        annotate_fontsize=9,
        overlay_offset=1.0,
        solver_alpha=solver_alpha,
        overhead_alpha=overhead_alpha,
    )

    plt.tight_layout()
    plt.savefig("Figure3_cdm27.pdf", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
