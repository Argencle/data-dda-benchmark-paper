import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgb


def add_legend_table(ax, code_colors, solver_alpha=0.95, overhead_alpha=0.35):
    """
    Petit tableau de légende :

    Lignes   : une par série (cles de code_colors)
    Colonnes : FFTs, Solver, Wall

    FFTs  : base_color (couleur saturée, matvec)
    Solver: lighten_color(base_color, 0.6) avec alpha solver_alpha
    Wall  : même couleur que solver avec alpha overhead_alpha
    """
    # Une ligne par série
    codes = list(code_colors.keys())
    col_labels = ["FFTs", "Solver", "Wall"]

    n_rows = len(codes)
    n_cols = len(col_labels)

    cell_text = [["" for _ in range(n_cols)] for _ in range(n_rows)]

    def rgba(c, a=1.0):
        r, g, b = to_rgb(c)
        return (r, g, b, a)

    cell_colours = [
        [(0.0, 0.0, 0.0, 0.0) for _ in range(n_cols)] for _ in range(n_rows)
    ]

    for i, code in enumerate(codes):
        base_color = code_colors[code]
        solver_col = lighten_color(base_color, 0.6)

        # FFTs (matvec) : saturé
        if "BLAS" not in code:
            cell_colours[i][0] = rgba(base_color, 1.0)
        # Solver
        cell_colours[i][1] = rgba(solver_col, solver_alpha)
        # Wall
        cell_colours[i][2] = rgba(solver_col, overhead_alpha)

    table = ax.table(
        cellText=cell_text,
        cellColours=cell_colours,
        rowLabels=codes,
        colLabels=col_labels,
        cellLoc="center",
        rowLoc="center",
        loc="upper left",
        bbox=[0.13, 0.78, 0.15, 0.20],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)

    for key, cell in table.get_celld().items():
        cell.set_linewidth(0.3)


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
# Grouped stacked bars
# ------------------------------------------------------------
def grouped_stacked_bars_ax(
    ax,
    title,
    x_labels,
    solver_series,
    total_series,
    matvec_series=None,  # e.g. {"OCL BiCG": [...], ...}
    code_colors=None,  # dict: series_name -> base color
    ylabel="Seconds",
    solver_alpha=0.95,
    overhead_alpha=0.35,
    annotate_fontsize=9,
    xtick_series=None,
):
    """
    Draw on a provided Axes `ax`:
      - grouped bars per (exe, solver) combo (OCL BiCG, OCL BLAS BiCGStab, etc.)
      - if no matvec for a series:
          bottom segment: solver time
          top segment   : overhead = total - solver
      - if matvec for a series:
          bottom : matvec time
          middle : solver - matvec
          top    : overhead = total - solver
    """
    series_names = list(solver_series.keys())
    assert set(series_names) == set(
        total_series.keys()
    ), "Solver/Total keys must match."
    n_series = len(series_names)
    n_groups = len(x_labels)

    # Sanity checks
    for s in series_names:
        assert len(solver_series[s]) == n_groups
        assert len(total_series[s]) == n_groups
        if matvec_series is not None and s in matvec_series:
            assert len(matvec_series[s]) == n_groups

    # Choose colors for each series
    if code_colors is None:
        default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        code_colors = {
            name: default_colors[i % len(default_colors)]
            for i, name in enumerate(series_names)
        }

    x = np.arange(n_groups)
    width = 0.8 / max(1, n_series)

    bar_positions = {}

    # Draw stacked bars for each series (combo exe/solver)
    for i, s in enumerate(series_names):
        base_color = code_colors[s]

        sol = np.array(solver_series[s], dtype=float)
        tot = np.array(total_series[s], dtype=float)

        sol_plot = np.nan_to_num(sol, nan=0.0)
        tot_plot = np.nan_to_num(tot, nan=0.0)
        over = np.maximum(tot_plot - sol_plot, 0.0)

        has_matvec = matvec_series is not None and s in matvec_series
        if has_matvec:
            mv = np.array(matvec_series[s], dtype=float)
            mv_plot = np.nan_to_num(mv, nan=0.0)
            mv_plot = np.minimum(mv_plot, sol_plot)  # safety
            sol_rest_plot = np.maximum(sol_plot - mv_plot, 0.0)
        else:
            mv = None
            mv_plot = None
            sol_rest_plot = sol_plot.copy()

        x_off = x + i * width - 0.4 + (width * n_series) / 2
        bar_positions[s] = x_off
        solver_color = lighten_color(base_color, 0.6)

        # --- Draw bars ---
        if has_matvec:
            # Matvec chunk (bottom)
            ax.bar(
                x_off,
                mv_plot,
                width,
                color=base_color,
                alpha=1.0,
            )
            # Remaining solver time
            ax.bar(
                x_off,
                sol_rest_plot,
                width,
                bottom=mv_plot,
                color=solver_color,
                alpha=solver_alpha,
            )
        else:
            # Whole solver chunk
            ax.bar(
                x_off,
                sol_plot,
                width,
                color=solver_color,
                alpha=solver_alpha,
            )

        # Overhead chunk
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
        for xi, si, oi, solver_val in zip(x_off, sol_plot, over, sol):
            if np.isfinite(solver_val) and si > 0:
                ax.text(
                    xi,
                    si,
                    f"{solver_val:.1f}",
                    ha="center",
                    va="bottom",
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
        # ------------------------------------------------------------
    # OUT OF MEMORY markers for RTX 2000 Ada
    # ------------------------------------------------------------
    oom_gpu_label = "RTX 2000 Ada"
    oom_series = ("OCL_BLAS BiCG",)
    if title == "n_x=250":
        if oom_gpu_label in x_labels:
            oom_idx = x_labels.index(oom_gpu_label)

            # hauteur visible basée sur les max des totaux
            max_total = 0.0
            for ss in series_names:
                arr = np.array(total_series[ss], dtype=float)
                arr = np.nan_to_num(arr, nan=0.0)
                if arr.size > 0:
                    max_total = max(max_total, float(np.max(arr)))

            if max_total <= 0:
                max_total = 1.0

            oom_h = 0.9 * max_total  # hauteur du rectangle
            oom_y = 0.0

            for ss in oom_series:
                xi = float(bar_positions[ss][oom_idx])

                rect = mpatches.Rectangle(
                    (xi - width * 0.5, oom_y),
                    width,
                    oom_h,
                    fill=False,
                    linewidth=1.2,
                    edgecolor="black",
                    zorder=7,
                )
                ax.add_patch(rect)

                ax.text(
                    xi,
                    oom_y + 0.5 * oom_h,
                    "OUT OF\nMEMORY",
                    ha="center",
                    va="center",
                    rotation=0,
                    fontsize=8,
                    zorder=8,
                )

    if xtick_series is not None and xtick_series in bar_positions:
        xtick_positions = bar_positions[xtick_series]
    else:
        xtick_positions = x

    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(x_labels, rotation=15, ha="right")
    ax.set_ylabel(ylabel)


# -------------------------------------------------------------------
# Helpers to load & label data
# -------------------------------------------------------------------
def load_adda_gpu_data():
    """
    Load all ADDA GPU CSVs, add a 'gpu_label' column, and concatenate.

    Files (à adapter si besoin) :
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


def get_adda_times_combo(df_adda, N, gpu_label, exe_name, solver_name):
    """
    Extract mean solver_time, total time, and matvec/FFT time
    for a given combination (exe, solver).

    Parameters
    ----------
    df_adda    : DataFrame
    N          : int
    gpu_label  : str
    exe_name   : str (e.g. 'adda_ocl', 'adda_ocl_blas')
    solver_name: str (e.g. 'bicg', 'bicgstab')

    Returns
    -------
    solver_mean, total_mean, matvec_mean
    """
    # On suppose une colonne 'solver' avec 'bicg' / 'bicgstab'
    # et une colonne 'exe' avec 'adda_ocl', 'adda_ocl_blas', etc.
    mask = (
        (df_adda["n"] == N)
        & (df_adda["gpu_label"] == gpu_label)
        & (df_adda["exe"].str.lower() == exe_name.lower())
        & (df_adda["solver"].str.lower() == solver_name.lower())
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
        matvec_cv_pct = (
            100 * matvec_std / matvec_mean if matvec_mean != 0 else np.nan
        )
        print(
            f"Mean ADDA matvec time: {matvec_mean:.1f} with std: {matvec_cv_pct:.2f}%"
        )
    else:
        matvec_mean = np.nan
        matvec_std = np.nan
        matvec_cv_pct = np.nan
    solver_std = sub["solver_time"].std(ddof=1)
    total_std = sub[total_col].std(ddof=1)
    solver_cv_pct = (
        100 * solver_std / solver_mean if solver_mean != 0 else np.nan
    )
    total_cv_pct = 100 * total_std / total_mean if total_mean != 0 else np.nan
    print(
        f"Mean ADDA solver time: {solver_mean:.1f} with std: {solver_cv_pct:.2f}%, total time: {total_mean:.1f} with std: {total_cv_pct:.2f}%"
    )
    with open("Appendix_mean_std_adda.txt", "a") as f:
        f.write(
            f"{N},{gpu_label},{exe_name},{solver_name},{matvec_mean:.1f},{matvec_cv_pct:.2f},{solver_mean:.1f},{solver_cv_pct:.2f},{total_mean:.1f},{total_cv_pct:.2f}\n"
        )
    if (matvec_cv_pct > 5.0) or (total_cv_pct > 5.0) or (solver_cv_pct > 5.0):
        print(sub)

    return solver_mean, total_mean, matvec_mean


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    # X labels for each N (sans RTX 6000 Ada)
    x_labels_150 = [
        "NVIDIA A100",
        "NVIDIA H200",
        "RTX 2000 Ada",
    ]
    x_labels_250 = [
        "NVIDIA A100",
        "NVIDIA H200",
        "RTX 2000 Ada",
    ]

    with open("Appendix_mean_std_adda.txt", "a") as f:
        f.write(
            "N,GPU,exe,solver,MatvecMean,MatvecCVpct,SolverMean,SolverCVpct,TotalMean,TotalCVpct\n"
        )

    df_adda = load_adda_gpu_data()

    # Combinaisons (exe, solver) -> label de série
    combos = {
        ("adda_ocl", "bicgstab"): "OCL BiCGStab",
        ("adda_ocl", "bicg"): "OCL BiCG",
        ("adda_ocl_blas", "bicg"): "OCL_BLAS BiCG",
    }

    series_names = list(combos.values())
    matvec_series_names = [name for name in series_names if "BLAS" not in name]

    # ----------------- N = 150 -----------------
    solver_150 = {name: [] for name in series_names}
    total_150 = {name: [] for name in series_names}
    matvec_150 = {name: [] for name in matvec_series_names}

    for gpu_label in x_labels_150:
        for (exe_name, solver_name), label in combos.items():
            sol, tot, mv = get_adda_times_combo(
                df_adda,
                N=150,
                gpu_label=gpu_label,
                exe_name=exe_name,
                solver_name=solver_name,
            )
            solver_150[label].append(sol)
            total_150[label].append(tot)
            if (
                label in matvec_150
            ):  # <--- on stocke mv seulement pour les non-BLAS
                matvec_150[label].append(mv)

    # ----------------- N = 250 -----------------
    solver_250 = {name: [] for name in series_names}
    total_250 = {name: [] for name in series_names}
    matvec_250 = {name: [] for name in matvec_series_names}

    for gpu_label in x_labels_250:
        for (exe_name, solver_name), label in combos.items():
            sol, tot, mv = get_adda_times_combo(
                df_adda,
                N=250,
                gpu_label=gpu_label,
                exe_name=exe_name,
                solver_name=solver_name,
            )
            solver_250[label].append(sol)
            total_250[label].append(tot)
            if (
                label in matvec_250
            ):  # <--- on stocke mv seulement pour les non-BLAS
                matvec_250[label].append(mv)

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

    # Couleurs de base pour chaque combo
    code_colors = {
        "OCL BiCGStab": "#1f77b4",
        "OCL BiCG": "#c44a12",
        "OCL_BLAS BiCG": "#708061",
    }

    # (a) N=150
    grouped_stacked_bars_ax(
        ax=ax_top,
        title=r"",
        x_labels=x_labels_150,
        solver_series=solver_150,
        total_series=total_150,
        matvec_series=matvec_150,
        code_colors=code_colors,
        ylabel="Seconds",
        annotate_fontsize=9,
        solver_alpha=solver_alpha,
        overhead_alpha=overhead_alpha,
        xtick_series="OCL_BLAS BiCG",
    )

    add_legend_table(
        ax_top,
        code_colors=code_colors,
        solver_alpha=solver_alpha,
        overhead_alpha=overhead_alpha,
    )

    # (b) N=250
    grouped_stacked_bars_ax(
        ax=ax_bot,
        title=r"n_x=250",
        x_labels=x_labels_250,
        solver_series=solver_250,
        total_series=total_250,
        matvec_series=matvec_250,
        code_colors=code_colors,
        ylabel="Seconds",
        annotate_fontsize=9,
        solver_alpha=solver_alpha,
        overhead_alpha=overhead_alpha,
        xtick_series="OCL_BLAS BiCG",
    )

    plt.tight_layout()
    plt.savefig("Figure_Appendix.pdf", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
