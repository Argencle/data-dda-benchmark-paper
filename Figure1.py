import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1. Input files
# ============================================================
ifdda_file = "logs_MPI_cluster/ifdda_results_sorted.csv"
adda_file = "logs_MPI_cluster/adda_results_sorted.csv"
ddscat_file = "logs_MPI_cluster/ddscat_results_sorted.csv"

# ============================================================
# 2. General parameters
# ============================================================
Ns = [150, 250]

# Canonical method labels (for plotting order & style)
method_labels = [
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

# Thresholds on coefficient of variation (to hide tiny errorbars)
CV_TIME_MIN = 0.05  # 5 % on wall time
CV_SPEED_MIN = 0.05  # 5 % on speedup

# ============================================================
# 3. Read CSV files and aggregate repetitions
# ============================================================

# Dictionary: (method_label, N, cores) -> list of wall-clock times (s)
times_dict = {}


def add_sample(method, N, ncores, t):
    """Register a single timing sample for (method, N, ncores)."""
    key = (method, int(N), int(ncores))
    times_dict.setdefault(key, []).append(float(t))


# ---------- 3.1 IFDDA: OMP = cores ----------
# Columns:
# JOBID,N,OMP,EXE,rep,first_residual,last_residual,Cext,
# elapsed_time,num_iterations,num_matvec,total_time,solver_time
idf = pd.read_csv(ifdda_file)

for _, row in idf.iterrows():
    N = row["N"]
    omp = row["OMP"]
    exe = row["EXE"]
    tsec = row["total_time"]

    if pd.isna(tsec):
        continue

    if exe == "ifdda_measure":
        method = "IFDDA (FFTW)"
    else:
        continue

    add_sample(method, N, omp, tsec)


# ---------- 3.2 ADDA: NP = cores ----------
# Columns:
# JOBID,N,NP,FFT,rep,first_residual,last_residual,Cext,Qext,
# elapsed_time,total_iterations,total_matvec,total_wall_time,solver_time,fft_time
ad = pd.read_csv(adda_file)

for _, row in ad.iterrows():
    N = row["N"]
    np_cores = row["NP"]  # number of MPI processes
    fft = row["FFT"]
    tsec = row["total_wall_time"]

    if pd.isna(tsec):
        continue

    if fft == "FFTW":
        method = "ADDA (FFTW)"
    elif fft == "GPFA":
        method = "ADDA (GPFA)"
    else:
        continue

    add_sample(method, N, np_cores, tsec)


# ---------- 3.3 DDSCAT: OMP = cores ----------
# Columns (example):
# JOBID,N,FFT,OMP,rep,first_residual,last_residual,Qext,
# elapsed_time,elapsed_seconds,num_iterations
dd = pd.read_csv(ddscat_file)

for _, row in dd.iterrows():
    N = row["N"]
    omp = row["OMP"]
    fft = row["FFT"]
    tsec = row["elapsed_seconds"]

    if pd.isna(tsec):
        continue

    if fft == "FFTMKL":
        method = "DDSCAT (MKL)"
    elif fft == "GPFAFT":
        method = "DDSCAT (GPFA)"
    else:
        continue

    add_sample(method, N, omp, tsec)


# ============================================================
# 4. Available cores per grid (avoid e.g. 250 cores for n_x=150)
# ============================================================
cores_per_N = {}
for N in Ns:
    cores_N = sorted({k[2] for k in times_dict.keys() if k[1] == N})
    cores_per_N[N] = np.array(cores_N, dtype=int)

print("Cores for n_x=150:", cores_per_N[150])
print("Cores for n_x=250:", cores_per_N[250])

# ============================================================
# 5. Build mean/std arrays + speedups
# ============================================================


def build_time_and_speedup_arrays(N, method, cores_all):
    """
    For a given (N, method), return:
      - times_mean: average wall time on cores_all (NaN if missing)
      - times_std : sample standard deviation (0 if a single repetition, NaN if missing)
      - speed_mean: mean speedup wrt 1-core (t1_mean / t_i)
      - speed_std : sample std of speedup
    """
    times_mean = np.full_like(cores_all, np.nan, dtype=float)
    times_std = np.full_like(cores_all, np.nan, dtype=float)

    # Reference 1-core samples (for this method and grid)
    key_1 = (method, N, 1)
    t1_samples = np.array(times_dict.get(key_1, []), dtype=float)

    if t1_samples.size == 0 or not np.isfinite(t1_samples).all():
        t1_mean = np.nan
    else:
        t1_mean = t1_samples.mean()

    speed_mean = np.full_like(cores_all, np.nan, dtype=float)
    speed_std = np.full_like(cores_all, np.nan, dtype=float)

    for i, c in enumerate(cores_all):
        key = (method, N, int(c))
        samples = np.array(times_dict.get(key, []), dtype=float)

        if samples.size == 0:
            continue  # remain NaN

        # Wall time statistics
        times_mean[i] = samples.mean()
        times_std[i] = samples.std(ddof=1) if samples.size > 1 else 0.0

        # Speedup statistics
        if np.isfinite(t1_mean) and t1_mean > 0:
            speed_samples = t1_mean / samples
            speed_mean[i] = speed_samples.mean()
            speed_std[i] = (
                speed_samples.std(ddof=1) if speed_samples.size > 1 else 0.0
            )

    return times_mean, times_std, speed_mean, speed_std


times_mean = {150: {}, 250: {}}
times_std = {150: {}, 250: {}}
speed_mean = {150: {}, 250: {}}
speed_std = {150: {}, 250: {}}

for N in Ns:
    cores_N = cores_per_N[N]
    for method in method_labels:
        t_mean, t_std, s_mean, s_std = build_time_and_speedup_arrays(
            N, method, cores_N
        )
        times_mean[N][method] = t_mean
        times_std[N][method] = t_std
        speed_mean[N][method] = s_mean
        speed_std[N][method] = s_std

# ============================================================
# 6. Matplotlib style
# ============================================================
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

# We'll store one plain line handle per method for the legend (line + marker only)
legend_lines = {}


# Helper to plot line + optional errorbars, returning the line handle
def plot_line_with_errorbars(ax, x, y, yerr, method):
    """
    Plot a line with markers for the averages, and overlay separate
    errorbars (without additional legend entry). Returns the line handle.
    """
    marker = marker_map.get(method, "o")
    color = color_map.get(method, None)

    # Main line + markers (this one goes to the legend)
    (line,) = ax.plot(
        x,
        y,
        marker=marker,
        color=color,
        linewidth=1.4,
        label=method,
    )

    # Error bars (no label, so they don't appear in the legend)
    if yerr is not None and np.any(np.isfinite(yerr)):
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="none",
            ecolor=color,
            capsize=3,
            linewidth=1.0,
        )

    return line


# ============================================================
# 7. (a) Wall time, n_x=150
# ============================================================
N = 150
cores_150 = cores_per_N[N]
for method in method_labels:
    t = times_mean[N][method]
    e = times_std[N][method]
    m = np.isfinite(t)

    if not np.any(m):
        continue

    # Coefficient of variation to hide tiny errorbars
    cv = e / t
    e_plot = np.where((cv > CV_TIME_MIN) & np.isfinite(cv), e, np.nan)

    line = plot_line_with_errorbars(
        ax_time_150,
        cores_150[m],
        t[m],
        e_plot[m],
        method,
    )
    legend_lines.setdefault(method, line)

ax_time_150.set_ylabel("Wall-clock time (s)")
ax_time_150.set_title(r"Grid $n_x=150$")
ax_time_150.set_xscale("log", base=5)
ax_time_150.set_yscale("log", base=5)
ax_time_150.set_xticks(cores_150)
ax_time_150.set_xticklabels([str(c) for c in cores_150])
ax_time_150.set_yticks([5, 25, 125, 500])
ax_time_150.set_yticklabels([5, 25, 125, 500])
ax_time_150.text(
    0.02, 0.05, "(a)", transform=ax_time_150.transAxes, fontsize=12
)

# ============================================================
# 8. (b) Wall time, n_x=250
# ============================================================
N = 250
cores_250 = cores_per_N[N]
for method in method_labels:
    t = times_mean[N][method]
    e = times_std[N][method]
    m = np.isfinite(t)

    if not np.any(m):
        continue

    cv = e / t
    e_plot = np.where((cv > CV_TIME_MIN) & np.isfinite(cv), e, np.nan)

    line = plot_line_with_errorbars(
        ax_time_250,
        cores_250[m],
        t[m],
        e_plot[m],
        method,
    )
    legend_lines.setdefault(method, line)

ax_time_250.set_title(r"Grid $n_x=250$")
ax_time_250.set_xscale("log", base=5)
ax_time_250.set_yscale("log", base=5)
ax_time_250.set_xticks(cores_250)
ax_time_250.set_xticklabels([str(c) for c in cores_250])
ax_time_250.set_ylabel("")
ax_time_250.set_yticks([25, 125, 625, 2500])
ax_time_250.set_yticklabels([25, 125, 625, 2500])
ax_time_250.text(
    0.02, 0.05, "(b)", transform=ax_time_250.transAxes, fontsize=12
)

# ============================================================
# 9. (c) Speedup, n_x=150
# ============================================================
N = 150
for method in method_labels:
    s = speed_mean[N][method]
    e = speed_std[N][method]
    m = np.isfinite(s)

    if not np.any(m):
        continue

    cv = e / s
    e_plot = np.where((cv > CV_SPEED_MIN) & np.isfinite(cv), e, np.nan)

    line = plot_line_with_errorbars(
        ax_speed_150,
        cores_150[m],
        s[m],
        e_plot[m],
        method,
    )
    legend_lines.setdefault(method, line)

# Ideal linear speedup (only on first few cores as in your previous script)
ideal_150 = cores_150 / cores_150[0]
ax_speed_150.plot(
    cores_150[:6],
    ideal_150[:6],
    linestyle="--",
    linewidth=1.0,
    color="black",
    label="Ideal (linear)",
)

ax_speed_150.set_ylabel("Speedup vs. 1 core")
ax_speed_150.set_xlabel("Cores")
ax_speed_150.set_xscale("log", base=5)
ax_speed_150.set_yscale("log", base=5)
ax_speed_150.set_xticks(cores_150)
ax_speed_150.set_xticklabels([str(c) for c in cores_150])
ax_speed_150.set_yticks([1, 5, 25])
ax_speed_150.set_yticklabels([1, 5, 25])
ax_speed_150.text(
    0.02, 0.90, "(c)", transform=ax_speed_150.transAxes, fontsize=12
)

# ============================================================
# 10. (d) Speedup, n_x=250
# ============================================================
N = 250
for method in method_labels:
    s = speed_mean[N][method]
    e = speed_std[N][method]
    m = np.isfinite(s)

    if not np.any(m):
        continue

    cv = e / s
    e_plot = np.where((cv > CV_SPEED_MIN) & np.isfinite(cv), e, np.nan)

    line = plot_line_with_errorbars(
        ax_speed_250,
        cores_250[m],
        s[m],
        e_plot[m],
        method,
    )
    legend_lines.setdefault(method, line)

ideal_250 = cores_250 / cores_250[0]
ax_speed_250.plot(
    cores_250[:7],
    ideal_250[:7],
    linestyle="--",
    linewidth=1.0,
    color="black",
    label="Ideal (linear)",
)

ax_speed_250.set_xlabel("Cores")
ax_speed_250.set_xscale("log", base=5)
ax_speed_250.set_yscale("log", base=5)
ax_speed_250.set_xticks(cores_250)
ax_speed_250.set_xticklabels([str(c) for c in cores_250])
ax_speed_250.set_ylabel("")
ax_speed_250.set_yticks([1, 5, 25])
ax_speed_250.set_yticklabels([1, 5, 25])
ax_speed_250.text(
    0.02, 0.90, "(d)", transform=ax_speed_250.transAxes, fontsize=12
)

# Remove x labels on top row
ax_time_150.set_xlabel("")
ax_time_250.set_xlabel("")

# ============================================================
# 11. Global legend and save
# ============================================================
# Use only the plain line+marker handles (no errorbars) in the legend
handles = [legend_lines[m] for m in method_labels if m in legend_lines]
labels = [m for m in method_labels if m in legend_lines]

fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=3,
    frameon=False,
    bbox_to_anchor=(0.5, -0.01),
)

fig.tight_layout(rect=(0, 0.03, 1, 1))

fig.savefig("Figure1.pdf", bbox_inches="tight", dpi=300)
plt.show()
