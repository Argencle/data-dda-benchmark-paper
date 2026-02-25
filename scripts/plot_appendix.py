import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgb

DATA = "datas/appendix_figure_data.csv"
OUTFIG = "figures/Figure_Appendix.pdf"

GPUS_150 = ["NVIDIA A100", "NVIDIA H200", "RTX 2000 Ada"]
GPUS_250 = ["NVIDIA A100", "NVIDIA H200", "RTX 2000 Ada"]

SERIES_ORDER = ["OCL BiCGStab", "OCL BiCG", "OCL_BLAS BiCG"]

CODE_COLORS = {
    "OCL BiCGStab": "#1f77b4",
    "OCL BiCG": "#c44a12",
    "OCL_BLAS BiCG": "#708061",
}

SOLVER_ALPHA = 0.95
OVERHEAD_ALPHA = 0.35


def lighten_color(color, factor=0.6):
    r, g, b = to_rgb(color)
    return (
        (1 - factor) + factor * r,
        (1 - factor) + factor * g,
        (1 - factor) + factor * b,
    )


def add_legend_table(ax):
    codes = SERIES_ORDER
    col_labels = ["FFTs", "Solver", "Wall"]

    cell_text = [["" for _ in col_labels] for _ in codes]
    cell_colours = [[(0, 0, 0, 0) for _ in col_labels] for _ in codes]

    for i, code in enumerate(codes):
        base = CODE_COLORS[code]
        solver_col = lighten_color(base, 0.6)

        # FFTs shown only for non-BLAS
        if "BLAS" not in code:
            cell_colours[i][0] = (*to_rgb(base), 1.0)

        cell_colours[i][1] = (*to_rgb(solver_col), SOLVER_ALPHA)
        cell_colours[i][2] = (*to_rgb(solver_col), OVERHEAD_ALPHA)

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
    for _, cell in table.get_celld().items():
        cell.set_linewidth(0.3)


def plot_panel(ax, dfN, gpus, title):
    x = np.arange(len(gpus))
    n_series = len(SERIES_ORDER)
    width = 0.8 / n_series

    bar_xpos = {}

    for i, s in enumerate(SERIES_ORDER):
        base = CODE_COLORS[s]
        solver_col = lighten_color(base, 0.6)

        sub = dfN[dfN["series"] == s].set_index("gpu")

        solver = np.array(
            [
                sub.loc[g, "solver_mean"] if g in sub.index else np.nan
                for g in gpus
            ],
            dtype=float,
        )
        total = np.array(
            [
                sub.loc[g, "total_mean"] if g in sub.index else np.nan
                for g in gpus
            ],
            dtype=float,
        )
        matvec = np.array(
            [
                sub.loc[g, "matvec_mean"] if g in sub.index else np.nan
                for g in gpus
            ],
            dtype=float,
        )

        solver_p = np.nan_to_num(solver, nan=0.0)
        total_p = np.nan_to_num(total, nan=0.0)

        overhead = np.maximum(total_p - solver_p, 0.0)

        has_matvec = "BLAS" not in s
        if has_matvec:
            matvec_p = np.nan_to_num(matvec, nan=0.0)
            matvec_p = np.minimum(matvec_p, solver_p)
            solver_rest = np.maximum(solver_p - matvec_p, 0.0)
        else:
            matvec_p = None
            solver_rest = solver_p

        x_off = x + i * width - 0.4 + (width * n_series) / 2
        bar_xpos[s] = x_off

        # draw bars
        if has_matvec:
            ax.bar(x_off, matvec_p, width, color=base, alpha=1.0)
            ax.bar(
                x_off,
                solver_rest,
                width,
                bottom=matvec_p,
                color=solver_col,
                alpha=SOLVER_ALPHA,
            )
        else:
            ax.bar(
                x_off, solver_p, width, color=solver_col, alpha=SOLVER_ALPHA
            )

        ax.bar(
            x_off,
            overhead,
            width,
            bottom=solver_p,
            color=solver_col,
            alpha=OVERHEAD_ALPHA,
        )

        # annotations (total on top, solver on solver height)
        for xi, sp, oh, tval, sval in zip(
            x_off, solver_p, overhead, total, solver
        ):
            if np.isfinite(tval) and tval > 0:
                ax.text(
                    xi,
                    sp + oh,
                    f"{tval:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
            if np.isfinite(sval) and sp > 0:
                ax.text(
                    xi, sp, f"{sval:.1f}", ha="center", va="bottom", fontsize=9
                )

        # matvec labels
        if has_matvec:
            for xi, mv in zip(x_off, matvec_p):
                if mv > 0:
                    ax.text(
                        xi,
                        mv / 2,
                        f"{mv:.1f}",
                        ha="center",
                        va="center",
                        fontsize=9,
                        color="black",
                    )

    # OOM marker for N=250, RTX 2000 Ada, OCL_BLAS BiCG
    if title == "n_x=250" and "RTX 2000 Ada" in gpus:
        idx = gpus.index("RTX 2000 Ada")
        xs = bar_xpos["OCL_BLAS BiCG"][idx]

        max_total = (
            float(np.nanmax(dfN["total_mean"].to_numpy()))
            if np.isfinite(dfN["total_mean"]).any()
            else 1.0
        )
        oom_h = 0.9 * max_total

        rect = mpatches.Rectangle(
            (xs - width * 0.5, 0.0),
            width,
            oom_h,
            fill=False,
            linewidth=1.2,
            edgecolor="black",
            zorder=7,
        )
        ax.add_patch(rect)
        ax.text(
            xs,
            0.5 * oom_h,
            "OUT OF\nMEMORY",
            ha="center",
            va="center",
            fontsize=8,
            zorder=8,
        )

    # axis labels
    ax.set_title(title)
    ax.set_ylabel("Seconds")
    ax.set_xticks(bar_xpos["OCL_BLAS BiCG"])
    ax.set_xticklabels(gpus, rotation=15, ha="right")


def main():
    df = pd.read_csv(DATA)

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

    plot_panel(ax_top, df[df["N"] == 150], GPUS_150, "n_x=150")
    add_legend_table(ax_top)

    plot_panel(ax_bot, df[df["N"] == 250], GPUS_250, "n_x=250")

    plt.tight_layout()
    plt.savefig(OUTFIG, dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
