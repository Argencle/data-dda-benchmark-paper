import numpy as np
import pandas as pd

OUT = "Figures/figure3_data.csv"

NS = [150, 250]
GPUS = ["NVIDIA A100", "NVIDIA H200", "RTX 6000 Ada", "RTX 2000 Ada"]

BASELINE_OMP = 1
OVERLAY_OMP = 10

IFDDA_SOURCES = [
    ("logs_GPU_cluster/ifdda_gpu_results_sorted.csv", "cluster", None),
    ("logs_GPU_6000Ada/ifdda_gpu_results.csv", "single", "RTX 6000 Ada"),
    (
        "logs_GPU_2000Ada/ifdda_gpu_results_sorted.csv",
        "single",
        "RTX 2000 Ada",
    ),
]

ADDA_SOURCES = [
    ("logs_GPU_cluster/adda_gpu_results_sorted.csv", "cluster", None),
    ("logs_GPU_6000Ada/adda_gpu_results.csv", "single", "RTX 6000 Ada"),
    ("logs_GPU_2000Ada/adda_gpu_results_sorted.csv", "single", "RTX 2000 Ada"),
]

PARTITION_TO_GPU = {
    "gpu_all": "NVIDIA A100",
    "gpu_h200": "NVIDIA H200",
}


def mean_std(series: pd.Series):
    """Return (mean, std)"""
    return float(series.mean()), float(series.std(ddof=1))


def load_ifdda_all():
    dfs = []
    for path, kind, gpu_label in IFDDA_SOURCES:
        df = pd.read_csv(path)
        df.columns = [c.lower() for c in df.columns]

        if kind == "cluster":
            df["gpu"] = df["partition"].map(PARTITION_TO_GPU)
        else:
            df["gpu"] = gpu_label

        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    #  Gpu count as 1 cpu so we replace to 1
    df["omp"] = df["omp"].replace(2, 1)

    return df


def load_adda_all():
    dfs = []
    for path, kind, gpu_label in ADDA_SOURCES:
        df = pd.read_csv(path)
        df.columns = [c.lower() for c in df.columns]

        if kind == "cluster":
            df["gpu"] = df["partition"].map(PARTITION_TO_GPU)
        else:
            df["gpu"] = gpu_label

        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    df["total"] = df["total_wall_time"]
    df["matvec"] = df["fft_time"]

    return df


def main():
    df_ifdda = load_ifdda_all()
    df_adda = load_adda_all()

    out_rows = []

    for N in NS:
        for gpu in GPUS:
            sub = df_adda[
                (df_adda["n"] == N)
                & (df_adda["gpu"] == gpu)
                & (df_adda["solver"].astype(str).str.lower() == "bicgstab")
                & (
                    df_adda["exe"]
                    .astype(str)
                    .str.lower()
                    .isin({"adda_ocl", "adda_ocl_blas"})
                )
            ]

            solver_mean, solver_std = mean_std(sub["solver_time"])
            total_mean, total_std = mean_std(sub["total"])
            matvec_mean, matvec_std = mean_std(sub["matvec"])

            out_rows.append(
                dict(
                    N=N,
                    gpu=gpu,
                    code="ADDA",
                    solver_mean=solver_mean,
                    solver_std=solver_std,
                    total_mean=total_mean,
                    total_std=total_std,
                    matvec_mean=matvec_mean,
                    matvec_std=matvec_std,
                    overlay_total_mean=np.nan,
                    overlay_total_std=np.nan,
                )
            )

    for N in NS:
        for gpu in GPUS:
            for precision, exe_name in [
                ("DP", "ifdda_gpu"),
                ("SP", "ifdda_gpu_sp"),
            ]:
                sub1 = df_ifdda[
                    (df_ifdda["n"] == N)
                    & (df_ifdda["gpu"] == gpu)
                    & (df_ifdda["exe"].astype(str) == exe_name)
                    & (df_ifdda["omp"] == BASELINE_OMP)
                ]

                sub10 = df_ifdda[
                    (df_ifdda["n"] == N)
                    & (df_ifdda["gpu"] == gpu)
                    & (df_ifdda["exe"].astype(str) == exe_name)
                    & (df_ifdda["omp"] == OVERLAY_OMP)
                ]

                solver_mean, solver_std = mean_std(sub1["solver_time"])
                total_mean, total_std = mean_std(sub1["total_time"])

                overlay_mean, overlay_std = mean_std(sub10["total_time"])

                out_rows.append(
                    dict(
                        N=N,
                        gpu=gpu,
                        code=f"IFDDA {precision}",
                        solver_mean=solver_mean,
                        solver_std=solver_std,
                        total_mean=total_mean,
                        total_std=total_std,
                        matvec_mean=np.nan,
                        matvec_std=np.nan,
                        overlay_total_mean=overlay_mean,
                        overlay_total_std=overlay_std,
                    )
                )

    out = pd.DataFrame(out_rows)
    out.to_csv(OUT, index=False)


if __name__ == "__main__":
    main()
