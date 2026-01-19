import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb

DATA = "Figures/figure3_data.csv"
OUTFIG = "Figures/Figure3.pdf"

CV_PRINT_THR = 0.05  # 5%


def add_legend_table(ax, code_colors, solver_alpha=0.95, overhead_alpha=0.35):
    """
    Legend table:

    Rows    : ADDA, IFDDA DP, IFDDA SP
    Columns : FFTs, Solver, Wall, 10-core
    """
    codes = ["ADDA", "IFDDA DP", "IFDDA SP"]
    col_labels = ["FFTs", "Solver", "Wall", "10-core"]

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
        base = code_colors[code]
        solver_col = lighten_color(base, 0.6)

        if code == "ADDA":
            cell_colours[i][0] = rgba(base, 1.0)

        cell_colours[i][1] = rgba(solver_col, solver_alpha)
        cell_colours[i][2] = rgba(solver_col, overhead_alpha)

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

    last_data_col = n_cols - 1

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


def lighten_color(color, factor=0.7):
    """Return a lighter shade of the given matplotlib color by mixing with white."""
    r, g, b = to_rgb(color)
    return (
        (1 - factor) + factor * r,
        (1 - factor) + factor * g,
        (1 - factor) + factor * b,
    )


def grouped_stacked_bars_with_overlay_ax(
    ax,
    title,
    x_labels,
    solver_series,
    total_series,
    overlay_total_series=None,
    matvec_series=None,
    code_colors=None,
    xtick_series="IFDDA SP",
    ylabel="Seconds",
    solver_alpha=0.95,
    overhead_alpha=0.35,
    annotate_fontsize=9,
    overlay_offset=0.6,
):
    series_names = list(solver_series.keys())
    assert set(series_names) == set(total_series.keys())

    n_series = len(series_names)
    n_groups = len(x_labels)

    x = np.arange(n_groups)
    width = 0.8 / max(1, n_series)

    if code_colors is None:
        default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        code_colors = {
            name: default_colors[i % len(default_colors)]
            for i, name in enumerate(series_names)
        }

    bar_positions = {}

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
            mv_plot = np.minimum(mv_plot, sol_plot)
            sol_rest_plot = np.maximum(sol_plot - mv_plot, 0.0)
        else:
            mv = None
            mv_plot = None
            sol_rest_plot = sol_plot.copy()

        x_off = x + i * width - 0.4 + (width * n_series) / 2
        bar_positions[s] = x_off

        solver_color = lighten_color(base_color, 0.6)

        if has_matvec and s == "ADDA":
            ax.bar(x_off, mv_plot, width, color=base_color, alpha=1.0)
            ax.bar(
                x_off,
                sol_rest_plot,
                width,
                bottom=mv_plot,
                color=solver_color,
                alpha=solver_alpha,
            )
        else:
            ax.bar(
                x_off,
                sol_plot,
                width,
                color=solver_color,
                alpha=solver_alpha,
            )

        ax.bar(
            x_off,
            over,
            width,
            bottom=sol_plot,
            color=solver_color,
            alpha=overhead_alpha,
        )

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

    if overlay_total_series:
        for s, arr in overlay_total_series.items():
            if s not in bar_positions:
                continue
            x_off = bar_positions[s]
            arr = np.array(arr, dtype=float)
            arr_plot = np.nan_to_num(arr, nan=0.0)

            for xi, yi, val in zip(x_off, arr_plot, arr):
                if np.isfinite(val) and yi > 0:
                    ax.hlines(
                        yi,
                        xi - width * 0.45,
                        xi + width * 0.45,
                        linewidth=3.0,
                        color="black",
                    )
                    ax.text(
                        xi,
                        yi + overlay_offset,
                        f"{val:.1f}",
                        ha="center",
                        va="bottom",
                        fontsize=annotate_fontsize,
                    )

    # OUT OF MEMORY marker
    oom_gpu_label = "RTX 2000 Ada"
    oom_series = ("IFDDA DP", "IFDDA SP")

    if title == r"n_x=250":
        if oom_gpu_label in x_labels:
            oom_idx = x_labels.index(oom_gpu_label)

            max_total = 0.0
            for s in series_names:
                arr = np.array(total_series[s], dtype=float)
                arr = np.nan_to_num(arr, nan=0.0)
                if arr.size > 0:
                    max_total = max(max_total, float(np.max(arr)))
            if max_total <= 0:
                max_total = 1.0

            oom_h = 0.9 * max_total
            oom_y = 0.0

            for s in oom_series:

                xi = float(bar_positions[s][oom_idx])

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

    if xtick_series is not None and xtick_series in bar_positions:
        xtick_positions = bar_positions[xtick_series]
    else:
        xtick_positions = x

    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(x_labels, rotation=15, ha="right")
    ax.set_ylabel(ylabel)


def print_cv_warnings(df, thr=0.05):
    pairs = [
        ("solver_mean", "solver_std"),
        ("total_mean", "total_std"),
        ("matvec_mean", "matvec_std"),
        ("overlay_total_mean", "overlay_total_std"),
    ]
    for mean_col, std_col in pairs:
        mean = pd.to_numeric(df[mean_col], errors="coerce")
        std = pd.to_numeric(df[std_col], errors="coerce")
        mask = mean.notna() & std.notna() & (mean != 0) & ((std / mean) > thr)
        if mask.any():
            print(f"\n[WARN] CV > {thr*100:.0f}% for {std_col}/{mean_col}:")
            print(df.loc[mask, ["N", "gpu", "code", mean_col, std_col]])


def main():
    df = pd.read_csv(DATA)

    # print if any std/mean > 5%
    print_cv_warnings(df, thr=CV_PRINT_THR)

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

    code_colors = {
        "ADDA": "#1f77b4",
        "IFDDA DP": "#ff7f0e",
        "IFDDA SP": "#2ca02c",
    }

    solver_alpha = 0.95
    overhead_alpha = 0.35

    def series_for(N, x_labels):
        dfN = df[df["N"] == N].copy()
        dfN = dfN.set_index(["gpu", "code"])

        solver = {k: [] for k in ["ADDA", "IFDDA DP", "IFDDA SP"]}
        total = {k: [] for k in ["ADDA", "IFDDA DP", "IFDDA SP"]}
        overlay = {k: [] for k in ["IFDDA DP", "IFDDA SP"]}
        matvec = {"ADDA": []}

        for gpu in x_labels:
            # ADDA
            s = dfN.loc[(gpu, "ADDA"), "solver_mean"]

            t = dfN.loc[(gpu, "ADDA"), "total_mean"]

            m = dfN.loc[(gpu, "ADDA"), "matvec_mean"]
            solver["ADDA"].append(float(s))
            total["ADDA"].append(float(t))
            matvec["ADDA"].append(float(m))

            # IFDDA DP
            s = dfN.loc[(gpu, "IFDDA DP"), "solver_mean"]
            t = dfN.loc[(gpu, "IFDDA DP"), "total_mean"]
            o = dfN.loc[(gpu, "IFDDA DP"), "overlay_total_mean"]
            solver["IFDDA DP"].append(float(s))
            total["IFDDA DP"].append(float(t))
            overlay["IFDDA DP"].append(float(o))

            # IFDDA SP
            s = dfN.loc[(gpu, "IFDDA SP"), "solver_mean"]
            t = dfN.loc[(gpu, "IFDDA SP"), "total_mean"]
            o = dfN.loc[(gpu, "IFDDA SP"), "overlay_total_mean"]
            solver["IFDDA SP"].append(float(s))
            total["IFDDA SP"].append(float(t))
            overlay["IFDDA SP"].append(float(o))

        return solver, total, overlay, matvec

    solver_150, total_150, overlay_150, matvec_150 = series_for(
        150, x_labels_150
    )
    solver_250, total_250, overlay_250, matvec_250 = series_for(
        250, x_labels_250
    )

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
        matvec_series=matvec_150,
        code_colors=code_colors,
        xtick_series="IFDDA SP",
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
    plt.savefig(OUTFIG, dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
