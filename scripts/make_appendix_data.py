import numpy as np
import pandas as pd

OUT = "datas/appendix_figure_data.csv"

SOURCES = [
    ("datas/logs_GPU_cluster/adda_gpu_results_sorted.csv", "cluster", None),
    (
        "datas/logs_GPU_2000Ada/adda_gpu_results_sorted.csv",
        "single",
        "RTX 2000 Ada",
    ),
]

PARTITION_TO_GPU = {
    "gpu_all": "NVIDIA A100",
    "gpu_h200": "NVIDIA H200",
}

COMBOS = {
    ("adda_ocl", "bicgstab"): "OCL BiCGStab",
    ("adda_ocl", "bicg"): "OCL BiCG",
    ("adda_ocl_blas", "bicg"): "OCL_BLAS BiCG",
}

NS = [150, 250]
GPUS = ["NVIDIA A100", "NVIDIA H200", "RTX 2000 Ada"]


def load_all_adda():
    dfs = []
    for path, kind, gpu_label in SOURCES:
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
    df = load_all_adda()

    out_rows = []

    for N in NS:
        for gpu in GPUS:
            for (exe, solver), series in COMBOS.items():
                sub = df[
                    (df["n"] == N)
                    & (df["gpu"] == gpu)
                    & (df["exe"].str.lower() == exe.lower())
                    & (df["solver"].str.lower() == solver.lower())
                ]

                solver_mean = pd.to_numeric(
                    sub["solver_time"], errors="coerce"
                ).mean()
                total_mean = pd.to_numeric(
                    sub["total"], errors="coerce"
                ).mean()
                matvec_mean = pd.to_numeric(
                    sub["matvec"], errors="coerce"
                ).mean()

                out_rows.append(
                    {
                        "N": N,
                        "gpu": gpu,
                        "series": series,
                        "solver_mean": solver_mean,
                        "total_mean": total_mean,
                        "matvec_mean": matvec_mean,
                    }
                )

    out = pd.DataFrame(out_rows)
    out.to_csv(OUT, index=False)


if __name__ == "__main__":
    main()
