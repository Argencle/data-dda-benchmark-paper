import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA = "Figures/figure1_data.csv"
OUTFIG = "Figures/Figure1.pdf"

METHOD_LABELS = [
    "DDSCAT (MKL)",
    "DDSCAT (GPFA)",
    "ADDA (FFTW)",
    "ADDA (GPFA)",
    "IFDDA (FFTW)",
]

marker_map = {
    "DDSCAT (MKL)": "o",
    "DDSCAT (GPFA)": "D",
    "ADDA (FFTW)": "v",
    "ADDA (GPFA)": "^",
    "IFDDA (FFTW)": "s",
}

color_map = {
    "DDSCAT (MKL)": "C4",
    "DDSCAT (GPFA)": "C3",
    "ADDA (FFTW)": "C1",
    "ADDA (GPFA)": "C0",
    "IFDDA (FFTW)": "C2",
}

CV_TIME_MIN = 0.05
CV_SPEED_MIN = 0.05


def plot_series(ax, sub, ycol, ecol, method, cv_thr):
    """Plot a curve (line+markers) + optional error bars (hidden if too small)."""
    sub = sub.sort_values("cores")
    x = sub["cores"].to_numpy()
    y = sub[ycol].to_numpy(dtype=float)
    e = sub[ecol].to_numpy(dtype=float)

    cv = e / y
    e_plot = np.where((cv > cv_thr) & np.isfinite(cv), e, np.nan)

    (line,) = ax.plot(
        x,
        y,
        marker=marker_map.get(method, "o"),
        color=color_map.get(method, None),
        linewidth=1.4,
        label=method,
    )
    if np.any(np.isfinite(e_plot)):
        ax.errorbar(
            x,
            y,
            yerr=e_plot,
            fmt="none",
            ecolor=color_map.get(method, None),
            capsize=3,
            linewidth=1.0,
        )
    return line


def main():
    df = pd.read_csv(DATA)

    plt.rcParams.update(
        {
            "figure.figsize": (7.0, 6.8),
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "lines.linewidth": 1.4,
            "lines.markersize": 5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(2, 2, sharex="col", sharey=False)
    (ax_time_150, ax_time_250), (ax_speed_150, ax_speed_250) = axes

    legend_lines = {}

    # ----- N=150 and N=250 blocks -----
    for N, ax_t, ax_s in [
        (150, ax_time_150, ax_speed_150),
        (250, ax_time_250, ax_speed_250),
    ]:
        dfN = df[df["N"] == N].copy()

        cores = np.sort(dfN["cores"].dropna().unique()).astype(int)

        # Time curves
        for method in METHOD_LABELS:
            sub = dfN[dfN["method"] == method]
            legend_lines.setdefault(
                method,
                plot_series(
                    ax_t, sub, "time_mean", "time_std", method, CV_TIME_MIN
                ),
            )

        ax_t.set_title(rf"Grid $n_x={N}$")
        ax_t.set_xscale("log", base=5)
        ax_t.set_yscale("log", base=5)
        ax_t.set_xticks(cores)
        ax_t.set_xticklabels([str(c) for c in cores])

        if N == 150:
            ax_t.set_ylabel("Wall-clock time (s)")
            ax_t.set_yticks([5, 25, 125, 500])
            ax_t.set_yticklabels([5, 25, 125, 500])
        else:
            ax_t.set_yticks([25, 125, 625, 2500])
            ax_t.set_yticklabels([25, 125, 625, 2500])

        # Speedup curves
        for method in METHOD_LABELS:
            sub = dfN[dfN["method"] == method]
            legend_lines.setdefault(
                method,
                plot_series(
                    ax_s, sub, "speed_mean", "speed_std", method, CV_SPEED_MIN
                ),
            )

        # Ideal speedup line
        ideal = cores / cores[0]
        k = 6 if N == 150 else 7
        ax_s.plot(
            cores[:k], ideal[:k], linestyle="--", linewidth=1.0, color="black"
        )

        ax_s.set_xscale("log", base=5)
        ax_s.set_yscale("log", base=5)
        ax_s.set_xticks(cores)
        ax_s.set_xticklabels([str(c) for c in cores])

        if N == 150:
            ax_s.set_ylabel("Speedup vs. 1 core")
            ax_s.set_xlabel("Cores")
            ax_s.set_yticks([1, 5, 25])
            ax_s.set_yticklabels([1, 5, 25])
        else:
            ax_s.set_xlabel("Cores")
            ax_s.set_yticks([1, 5, 25])
            ax_s.set_yticklabels([1, 5, 25])

    # Panel labels
    ax_time_150.text(
        0.02, 0.05, "(a)", transform=ax_time_150.transAxes, fontsize=12
    )
    ax_time_250.text(
        0.02, 0.05, "(b)", transform=ax_time_250.transAxes, fontsize=12
    )
    ax_speed_150.text(
        0.02, 0.90, "(c)", transform=ax_speed_150.transAxes, fontsize=12
    )
    ax_speed_250.text(
        0.02, 0.90, "(d)", transform=ax_speed_250.transAxes, fontsize=12
    )

    # Remove x labels on top row
    ax_time_150.set_xlabel("")
    ax_time_250.set_xlabel("")

    # Global legend (keep only line+marker handles)
    handles = [legend_lines[m] for m in METHOD_LABELS if m in legend_lines]
    labels = [m for m in METHOD_LABELS if m in legend_lines]
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUTFIG, bbox_inches="tight", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
