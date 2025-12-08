import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb


def add_color_legend_grid(
    ax, code_colors, solver_alpha=0.95, overhead_alpha=0.35
):
    """
    Ajoute dans le coin haut gauche de `ax` une légende organisée en tableau :

        lignes   : ADDA, IFDDA DP, IFDDA SP
        colonnes : FFTs, Solver, Wall, 10-core

    - FFTs  : couleur base (ADDA uniquement)
    - Solver: couleur solver (lighten_color) avec alpha élevé
    - Wall  : même couleur mais alpha plus faible
    - 10-core: ligne fine pour montrer que c'est un trait, pas une barre.
    """
    codes = ["ADDA", "IFDDA DP", "IFDDA SP"]
    col_labels = ["FFTs", "Solver", "Wall", "10-core"]

    # Position globale dans l'axe (coordonnées 0–1)
    x0, y0 = 0.1, 0.9  # coin haut gauche de la "table"
    col_w, row_h = 0.08, 0.04  # largeur d'une colonne / hauteur d'une ligne

    # --- Noms des colonnes au-dessus ---
    for j, label in enumerate(col_labels):
        xc = x0 + (j + 0.4) * col_w
        ax.text(
            xc,
            y0 + 0.03,
            label,
            ha="center",
            va="top",
            transform=ax.transAxes,
            fontsize=9,
        )

    # --- Lignes (codes) ---
    for i, code in enumerate(codes):
        # y de la ligne i
        y = y0 - (i + 1) * row_h

        # label de ligne (ADDA / IFDDA DP / IFDDA SP)
        ax.text(
            x0 - 0.01,
            y + 0.4 * row_h,
            code,
            ha="right",
            va="center",
            transform=ax.transAxes,
            fontsize=9,
        )

        base = code_colors[code]
        solver_col = lighten_color(base, 0.6)

        # Petite fonction utilitaire pour dessiner un rectangle "échantillon"
        def draw_box(col_idx, facecolor=None, alpha=1.0, edge=True):
            if facecolor is None:
                return
            box_x = x0 + col_idx * col_w + 0.02 * col_w
            box_y = y + 0.2 * row_h
            rect = mpatches.Rectangle(
                (box_x, box_y),
                0.76 * col_w,
                0.6 * row_h,
                transform=ax.transAxes,
                facecolor=facecolor,
                edgecolor="black" if edge else "none",
                linewidth=0.0,
                alpha=alpha,
                zorder=5,
            )
            ax.add_patch(rect)

        # Colonne FFTs : seulement pour ADDA
        if code == "ADDA":
            draw_box(col_idx=0, facecolor=base, alpha=1.0)

        # Colonne Solver
        draw_box(col_idx=1, facecolor=solver_col, alpha=solver_alpha)

        # Colonne Wall (overhead, plus transparent)
        draw_box(col_idx=2, facecolor=solver_col, alpha=overhead_alpha)

        # Colonne 10-core : petite ligne fine pour symboliser le trait
        x_line_start = x0 + 3 * col_w + 0.08 * col_w
        x_line_end = x0 + 3 * col_w + 0.75 * col_w
        y_line = y + 0.5 * row_h
        line = Line2D(
            [x_line_start, x_line_end],
            [y_line, y_line],
            transform=ax.transAxes,
            color="black",
            linewidth=3.0,  # plus fin pour bien marquer que c'est une ligne
            zorder=6,
        )
        ax.add_line(line)


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

    # --- X ticks: align labels with the chosen series (e.g. "IFDDA SP") ---
    if xtick_series is not None and xtick_series in bar_positions:
        xtick_positions = bar_positions[xtick_series]
    else:
        # fallback: group centers
        xtick_positions = x

    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(x_labels, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, pad=6)


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
        exe_names = ["ifdda_gpu", "ifdda_GPU"]
    else:
        exe_names = ["ifdda_gpu_sp", "ifdda_GPU_single"]

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
    else:
        matvec_mean = np.nan

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
    ]

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
    )

    add_color_legend_grid(
        ax_top,
        code_colors,
        solver_alpha=solver_alpha,
        overhead_alpha=overhead_alpha,
    )

    # (b) N=250
    grouped_stacked_bars_with_overlay_ax(
        ax=ax_bot,
        title=r"",
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
    )

    plt.tight_layout()
    plt.savefig("Figure3.pdf", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
