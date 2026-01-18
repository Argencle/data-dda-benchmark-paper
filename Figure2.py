import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter
from matplotlib.lines import Line2D

# ============================================================
# 1. Input files
# ============================================================
ddscat_file = "logs_MPI_laptop/ddscat_results_sorted.csv"
adda_file = "logs_MPI_laptop/adda_results_sorted.csv"
ifdda_file = "logs_MPI_laptop/ifdda_results_sorted.csv"

# Grid sizes
Ns = [150, 250]

# Core counts to consider (same for all methods)
cores = np.array([1, 2, 5, 10, 15, 22], dtype=int)

# Canonical labels for each method
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

# ============================================================
# 2. Read CSV files and aggregate repetitions
# ============================================================

# Dictionary: (method_label, N, cores) -> list of wall-clock times [s]
times_dict = {}


def add_sample(method, N, ncores, t):
    """Append one timing sample to the global dictionary."""
    key = (method, int(N), int(ncores))
    times_dict.setdefault(key, []).append(float(t))


# ---------- 2.1 DDSCAT ----------
# Columns: N,OMP,FFT,rep,first_residual,last_residual,Cext,Qext,
#          elapsed_time,total_iterations,total_matvec,total_wall_time,
#          solver_time,fft_time
dd = pd.read_csv(ddscat_file)

for _, row in dd.iterrows():
    N = row["N"]
    np_cores = row["OMP"]
    fft = row["FFT"]
    total_wall_time = row["elapsed_seconds"]

    if fft == "FFTMKL":
        method = "DDSCAT (MKL)"
    elif fft == "GPFAFT":
        method = "DDSCAT (GPFA)"
    else:
        continue

    add_sample(method, N, np_cores, total_wall_time)


# ---------- 2.2 ADDA ----------
# Columns: N,FFT,NP,rep,first_residual,last_residual,Qext,
#          elapsed_time,elapsed_seconds,num_iterations
ad = pd.read_csv(adda_file)

for _, row in ad.iterrows():
    N = row["N"]
    np_cores = row["NP"]  # effective number of cores
    fft = row["FFT"]
    tsec = row["total_wall_time"]

    if fft == "FFTW":
        method = "ADDA (FFTW)"
    elif fft == "GPFA":
        method = "ADDA (GPFA)"
    else:
        continue

    add_sample(method, N, np_cores, tsec)


# ---------- 2.3 IFDDA ----------
# Columns: N,OMP,EXE,rep,first_residual,last_residual,Cext,
#          elapsed_time,num_iterations,num_matvec,total_time,solver_time
idf = pd.read_csv(ifdda_file)

for _, row in idf.iterrows():
    N = row["N"]
    omp = row["OMP"]
    exe = row["EXE"]
    tsec = row["total_time"]

    # Here you keep only the FFTW (measure) flavor
    if exe == "ifdda":
        method = "IFDDA (FFTW)"
    else:
        continue

    add_sample(method, N, omp, tsec)


# ============================================================
# 3. Build mean/std arrays for time and speedup
# ============================================================


def build_time_and_speedup_arrays(N, method, cores_all):
    """
    For a given (N, method) pair, return:
      - times_mean : mean wall-clock time for each core count (NaN if missing)
      - times_std  : std dev of time (0 if single sample, NaN if missing)
      - speed_mean : mean speedup t(1 core)/t(n) for each core count
      - speed_std  : std dev of speedup
    """
    times_mean = np.full_like(cores_all, np.nan, dtype=float)
    times_std = np.full_like(cores_all, np.nan, dtype=float)

    # Reference time on 1 core (needed for speedup)
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
            continue  # stays NaN

        # Time statistics
        times_mean[i] = samples.mean()
        times_std[i] = samples.std(ddof=1) if samples.size > 1 else 0.0

        # Speedup statistics: t(1 core) / t(n)
        if np.isfinite(t1_mean) and t1_mean > 0.0:
            speed_samples = t1_mean / samples
            speed_mean[i] = speed_samples.mean()
            speed_std[i] = (
                speed_samples.std(ddof=1) if speed_samples.size > 1 else 0.0
            )

    return times_mean, times_std, speed_mean, speed_std


# Containers for N=150 and N=250
times_mean = {150: {}, 250: {}}
times_std = {150: {}, 250: {}}
speed_mean = {150: {}, 250: {}}
speed_std = {150: {}, 250: {}}

for N in Ns:
    for method in method_labels:
        t_mean, t_std, s_mean, s_std = build_time_and_speedup_arrays(
            N, method, cores
        )
        times_mean[N][method] = t_mean
        times_std[N][method] = t_std
        speed_mean[N][method] = s_mean
        speed_std[N][method] = s_std


# ============================================================
# 4. Matplotlib style
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

# Coefficient of variation thresholds: only show errorbars if std/mean > 5%
CV_TIME_MIN = 0.05  # 5%
CV_SPEED_MIN = 0.05  # 5%

fig, axes = plt.subplots(2, 2, sharex="col", sharey=False)
(ax_time_150, ax_time_250), (ax_speed_150, ax_speed_250) = axes

with open("Figure2_time_mean_std.txt", "a") as log_file:
    log_file.write("N, method, t_mean, t_std\n")
with open("Figure2_speed_mean_std.txt", "a") as log_file:
    log_file.write("N, method, s_mean, s_std\n")
# ============================================================
# 5. Panel (a): Wall time, N=150
# ============================================================
N = 150
for method in method_labels:
    t = times_mean[N][method]
    e = times_std[N][method]
    m = np.isfinite(t)

    if np.any(m):
        cv = e / t
        # Only show errorbars if relative std > CV_TIME_MIN
        e_plot = np.where((cv > CV_TIME_MIN) & np.isfinite(cv), e, np.nan)
        ax_time_150.errorbar(
            cores[m],
            t[m],
            yerr=e_plot[m],
            marker=marker_map.get(method, "o"),
            color=color_map.get(method, None),
            label=method,
            capsize=3,
            linewidth=1.4,
        )
    with open("Figure2_time_mean_std.txt", "a") as log_file:
        log_file.write(f"{N}, {method}, {t}, {e}\n")

ax_time_150.set_ylabel("Wall-clock time (s)")
ax_time_150.set_title(r"Grid $n_x=150$")
ax_time_150.set_xscale("log", base=5)
ax_time_150.set_yscale("log", base=5)
ax_time_150.set_xticks(cores)
ax_time_150.set_xticklabels([str(c) for c in cores])
ax_time_150.set_yticks([50, 150, 500])
ax_time_150.yaxis.set_major_locator(FixedLocator([50, 150, 500]))
ax_time_150.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f}"))
ax_time_150.text(
    0.02, 0.05, "(a)", transform=ax_time_150.transAxes, fontsize=12
)

# ============================================================
# 6. Panel (b): Wall time, N=250
# ============================================================
N = 250
for method in method_labels:
    t = times_mean[N][method]
    e = times_std[N][method]
    m = np.isfinite(t)

    if np.any(m):
        cv = e / t
        e_plot = np.where((cv > CV_TIME_MIN) & np.isfinite(cv), e, np.nan)
        ax_time_250.errorbar(
            cores[m],
            t[m],
            yerr=e_plot[m],
            marker=marker_map.get(method, "o"),
            color=color_map.get(method, None),
            label=method,
            capsize=3,
            linewidth=1.4,
        )
    with open("Figure2_time_mean_std.txt", "a") as log_file:
        log_file.write(f"{N}, {method}, {t}, {e}\n")

ax_time_250.set_title(r"Grid $n_x=250$")
ax_time_250.set_xscale("log", base=5)
ax_time_250.set_yscale("log", base=5)
ax_time_250.set_xticks(cores)
ax_time_250.set_xticklabels([str(c) for c in cores])
ax_time_250.set_ylabel("")
ax_time_250.set_yticks([250, 500, 2500])
ax_time_250.yaxis.set_major_locator(FixedLocator([250, 500, 2500]))
ax_time_250.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f}"))
ax_time_250.text(
    0.02, 0.05, "(b)", transform=ax_time_250.transAxes, fontsize=12
)

# ============================================================
# 7. Panel (c): Speedup, N=150
# ============================================================
N = 150
for method in method_labels:
    s = speed_mean[N][method]
    e = speed_std[N][method]
    m = np.isfinite(s)

    if np.any(m):
        cv = e / s
        e_plot = np.where((cv > CV_SPEED_MIN) & np.isfinite(cv), e, np.nan)
        ax_speed_150.errorbar(
            cores[m],
            s[m],
            yerr=e_plot[m],
            marker=marker_map.get(method, "o"),
            color=color_map.get(method, None),
            label=method,
            capsize=3,
            linewidth=1.4,
        )
    with open("Figure2_speed_mean_std.txt", "a") as log_file:
        log_file.write(f"{N}, {method}, {s}, {e}\n")

ideal_150 = cores / cores[0]
ax_speed_150.plot(
    cores[:3],
    ideal_150[:3],
    linestyle="--",
    linewidth=1.0,
    color="black",
    label="Ideal (linear)",
)

ax_speed_150.set_ylabel("Speedup vs. 1 core")
ax_speed_150.set_xlabel("Cores")
ax_speed_150.set_xscale("log", base=5)
ax_speed_150.set_yscale("log", base=5)
ax_speed_150.set_xticks(cores)
ax_speed_150.set_xticklabels([str(c) for c in cores])
ax_speed_150.set_yticks([1, 5])
ax_speed_150.yaxis.set_major_locator(FixedLocator([1, 5]))
ax_speed_150.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
ax_speed_150.text(
    0.02, 0.90, "(c)", transform=ax_speed_150.transAxes, fontsize=12
)

# ============================================================
# 8. Panel (d): Speedup, N=250
# ============================================================
N = 250
for method in method_labels:
    s = speed_mean[N][method]
    e = speed_std[N][method]
    m = np.isfinite(s)

    if np.any(m):
        cv = e / s
        e_plot = np.where((cv > CV_SPEED_MIN) & np.isfinite(cv), e, np.nan)
        ax_speed_250.errorbar(
            cores[m],
            s[m],
            yerr=e_plot[m],
            marker=marker_map.get(method, "o"),
            color=color_map.get(method, None),
            label=method,
            capsize=3,
            linewidth=1.4,
        )
    with open("Figure2_speed_mean_std.txt", "a") as log_file:
        log_file.write(f"{N}, {method}, {s}, {e}\n")

ideal_250 = cores / cores[0]
ax_speed_250.plot(
    cores[:3],
    ideal_250[:3],
    linestyle="--",
    linewidth=1.0,
    color="black",
    label="Ideal (linear)",
)

ax_speed_250.set_xlabel("Cores")
ax_speed_250.set_xscale("log", base=5)
ax_speed_250.set_yscale("log", base=5)
ax_speed_250.set_xticks(cores)
ax_speed_250.set_xticklabels([str(c) for c in cores])
ax_speed_250.set_ylabel("")
ax_speed_250.set_yticks([1, 5])
ax_speed_250.yaxis.set_major_locator(FixedLocator([1, 5]))
ax_speed_250.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
ax_speed_250.text(
    0.02, 0.90, "(d)", transform=ax_speed_250.transAxes, fontsize=12
)

# Remove x-labels on the top row
ax_time_150.set_xlabel("")
ax_time_250.set_xlabel("")

# Force x-limits so that the visible range starts at 1
for ax in [ax_time_150, ax_time_250, ax_speed_150, ax_speed_250]:
    ax.set_xlim(left=cores[0] - 0.15, right=cores[-1] * 1.2)

# ============================================================
# 9. Common legend with line + marker (no errorbar caps)
# ============================================================
proxy_handles = []
proxy_labels = []

for method in method_labels:
    proxy_handles.append(
        Line2D(
            [],
            [],
            marker=marker_map.get(method, "o"),
            linestyle="-",  # line + marker in legend
            color=color_map.get(method, None),
            markersize=5,
            linewidth=1.4,
        )
    )
    proxy_labels.append(method)

fig.legend(
    proxy_handles,
    proxy_labels,
    loc="lower center",
    ncol=3,
    frameon=False,
    bbox_to_anchor=(0.5, -0.01),
)

fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig("Figure2.pdf", bbox_inches="tight", dpi=300)

plt.show()
