import pandas as pd

JOBS = [
    # GPU 2000 Ada
    (
        "datas/logs_GPU_2000Ada/adda_gpu_results.csv",
        "datas/logs_GPU_2000Ada/adda_gpu_results_sorted.csv",
        ["N", "solver", "exe", "rep"],
    ),
    (
        "datas/logs_GPU_2000Ada/ifdda_gpu_results.csv",
        "datas/logs_GPU_2000Ada/ifdda_gpu_results_sorted.csv",
        ["N", "OMP", "exe", "rep"],
    ),
    # GPU cluster
    (
        "datas/logs_GPU_cluster/adda_gpu_results.csv",
        "datas/logs_GPU_cluster/adda_gpu_results_sorted.csv",
        ["N", "solver", "exe", "partition", "rep"],
    ),
    (
        "datas/logs_GPU_cluster/ifdda_gpu_results.csv",
        "datas/logs_GPU_cluster/ifdda_gpu_results_sorted.csv",
        ["N", "exe", "partition", "rep"],
    ),
    # MPI cluster
    (
        "datas/logs_MPI_cluster/adda_results.csv",
        "datas/logs_MPI_cluster/adda_results_sorted.csv",
        ["N", "NP", "FFT", "rep"],
    ),
    (
        "datas/logs_MPI_cluster/ifdda_results.csv",
        "datas/logs_MPI_cluster/ifdda_results_sorted.csv",
        ["N", "OMP", "rep"],
    ),
    (
        "datas/logs_MPI_cluster/ddscat_results.csv",
        "datas/logs_MPI_cluster/ddscat_results_sorted.csv",
        ["N", "OMP", "FFT", "rep"],
    ),
    # MPI laptop
    (
        "datas/logs_MPI_laptop/adda_results.csv",
        "datas/logs_MPI_laptop/adda_results_sorted.csv",
        ["N", "NP", "FFT", "rep"],
    ),
    (
        "datas/logs_MPI_laptop/ifdda_results.csv",
        "datas/logs_MPI_laptop/ifdda_results_sorted.csv",
        ["N", "OMP", "rep"],
    ),
    (
        "datas/logs_MPI_laptop/ddscat_results.csv",
        "datas/logs_MPI_laptop/ddscat_results_sorted.csv",
        ["N", "OMP", "FFT", "rep"],
    ),
]

for csv_path, sorted_csv_path, sort_cols in JOBS:
    df = pd.read_csv(csv_path)
    df_sorted = df.sort_values(by=sort_cols)
    df_sorted.to_csv(sorted_csv_path, index=False)
    print(f"Sorted CSV written to: {sorted_csv_path}")
