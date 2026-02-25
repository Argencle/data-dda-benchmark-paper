import numpy as np
import pandas as pd

IFDDA_FILE = "datas/logs_MPI_cluster/ifdda_results_filled.csv"
ADDA_FILE = "datas/logs_MPI_cluster/adda_results_filled.csv"
DDSCAT_FILE = "datas/logs_MPI_cluster/ddscat_results_filled.csv"

OUT = "datas/figure1_data.csv"


def main():
    rows = []

    ifdda = pd.read_csv(IFDDA_FILE)
    ifdda = ifdda[ifdda["EXE"] == "ifdda"].copy()
    ifdda = ifdda.rename(columns={"OMP": "cores", "total_time": "time_s"})
    ifdda["method"] = "IFDDA (FFTW)"
    rows.append(ifdda[["N", "cores", "method", "time_s"]])

    adda = pd.read_csv(ADDA_FILE)
    adda = adda.rename(columns={"NP": "cores", "total_wall_time": "time_s"})
    adda = adda[adda["FFT"].isin(["FFTW", "GPFA"])].copy()
    adda["method"] = np.where(
        adda["FFT"] == "FFTW", "ADDA (FFTW)", "ADDA (GPFA)"
    )
    rows.append(adda[["N", "cores", "method", "time_s"]])

    ddscat = pd.read_csv(DDSCAT_FILE)
    ddscat = ddscat.rename(
        columns={"OMP": "cores", "elapsed_seconds": "time_s"}
    )
    ddscat = ddscat[ddscat["FFT"].isin(["FFTMKL", "GPFAFT"])].copy()
    ddscat["method"] = np.where(
        ddscat["FFT"] == "FFTMKL", "DDSCAT (MKL)", "DDSCAT (GPFA)"
    )
    rows.append(ddscat[["N", "cores", "method", "time_s"]])

    # Merge all codes together
    raw = pd.concat(rows, ignore_index=True)

    # Compute mean/std of wall time for each (N, cores, method)
    time_mean = (
        raw.groupby(["N", "cores", "method"])["time_s"]
        .mean()
        .reset_index(name="time_mean")
    )

    time_std = (
        raw.groupby(["N", "cores", "method"])["time_s"]
        .std(ddof=1)
        .reset_index(name="time_std")
    )

    out = time_mean.merge(time_std, on=["N", "cores", "method"], how="left")

    # Compute speedup statistics for each (N, cores, method)

    # mean time at 1 core, for each (N, method)
    t1_mean = (
        raw[raw["cores"] == 1]
        .groupby(["N", "method"])["time_s"]
        .mean()
        .reset_index(name="t1_mean")
    )

    # attach t1_mean to every row, then compute speed = t1_mean / time_s
    raw = raw.merge(t1_mean, on=["N", "method"], how="left")
    raw["speed"] = raw["t1_mean"] / raw["time_s"]

    speed_mean = (
        raw.groupby(["N", "cores", "method"])["speed"]
        .mean()
        .reset_index(name="speed_mean")
    )

    speed_std = (
        raw.groupby(["N", "cores", "method"])["speed"]
        .std(ddof=1)
        .reset_index(name="speed_std")
    )

    out = out.merge(speed_mean, on=["N", "cores", "method"], how="left")
    out = out.merge(speed_std, on=["N", "cores", "method"], how="left")

    out.to_csv(OUT, index=False)


if __name__ == "__main__":
    main()
