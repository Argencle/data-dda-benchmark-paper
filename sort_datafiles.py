import pandas as pd

# GPU 2000 Ada results sorting

CSV_PATH = "logs_GPU_2000Ada/adda_gpu_results.csv"
SORTED_CSV_PATH = "logs_GPU_2000Ada/adda_gpu_results_sorted.csv"

df = pd.read_csv(CSV_PATH)

# Sort: N -> solver -> exe (custom) -> rep
df_sorted = df.sort_values(by=["N", "solver", "exe", "rep"])

df_sorted.to_csv(SORTED_CSV_PATH, index=False)
print(f"Sorted CSV written to: {SORTED_CSV_PATH}")


CSV_PATH = "logs_GPU_2000Ada/ifdda_gpu_results.csv"
SORTED_CSV_PATH = "logs_GPU_2000Ada/ifdda_gpu_results_sorted.csv"

df = pd.read_csv(CSV_PATH)

# Sort: N -> OMP -> exe (custom) -> rep
df_sorted = df.sort_values(
    by=[
        "N",
        "OMP",
        "exe",
        "rep",
    ]
)

df_sorted.to_csv(SORTED_CSV_PATH, index=False)
print(f"Sorted CSV written to: {SORTED_CSV_PATH}")


# GPU cluster results sorting

CSV_PATH = "logs_GPU_cluster/adda_gpu_results.csv"
SORTED_CSV_PATH = "logs_GPU_cluster/adda_gpu_results_sorted.csv"

df = pd.read_csv(CSV_PATH)

# Sort: N -> OMP -> exe (custom) -> rep
df_sorted = df.sort_values(
    by=[
        "N",
        "solver",
        "exe",
        "partition",
        "rep",
    ]
)

df_sorted.to_csv(SORTED_CSV_PATH, index=False)
print(f"Sorted CSV written to: {SORTED_CSV_PATH}")


# MPI cluster results sorting

CSV_PATH = "logs_MPI_cluster/adda_results.csv"
SORTED_CSV_PATH = "logs_MPI_cluster/adda_results_sorted.csv"

df = pd.read_csv(CSV_PATH)

# Sort: N -> OMP -> exe (custom) -> rep
df_sorted = df.sort_values(by=["N", "NP", "FFT", "rep"])

# Drop helper column and save sorted CSV
df_sorted.to_csv(SORTED_CSV_PATH, index=False)
print(f"Sorted CSV written to: {SORTED_CSV_PATH}")

CSV_PATH = "logs_MPI_cluster/ifdda_results.csv"
SORTED_CSV_PATH = "logs_MPI_cluster/ifdda_results_sorted.csv"

df = pd.read_csv(CSV_PATH)

# Sort: N -> OMP -> EXE (custom) -> rep
df_sorted = df.sort_values(by=["N", "OMP", "rep"])

df_sorted.to_csv(SORTED_CSV_PATH, index=False)
print(f"Sorted CSV written to: {SORTED_CSV_PATH}")

CSV_PATH = "logs_MPI_cluster/ddscat_results_filled.csv"
SORTED_CSV_PATH = "logs_MPI_cluster/ddscat_results_sorted.csv"

df = pd.read_csv(CSV_PATH)

# Sort: N -> OMP -> exe (custom) -> rep
df_sorted = df.sort_values(by=["N", "OMP", "FFT", "rep"])

# Drop helper column and save sorted CSV
df_sorted.to_csv(SORTED_CSV_PATH, index=False)
print(f"Sorted CSV written to: {SORTED_CSV_PATH}")

# MPI lap results sorting

CSV_PATH = "logs_MPI_laptop/adda_results.csv"
SORTED_CSV_PATH = "logs_MPI_laptop/adda_results_sorted.csv"


df = pd.read_csv(CSV_PATH)

# Sort: N -> OMP -> exe (custom) -> rep
df_sorted = df.sort_values(by=["N", "NP", "FFT", "rep"])

# Drop helper column and save sorted CSV
df_sorted.to_csv(SORTED_CSV_PATH, index=False)
print(f"Sorted CSV written to: {SORTED_CSV_PATH}")


CSV_PATH = "logs_MPI_laptop/ifdda_results.csv"
SORTED_CSV_PATH = "logs_MPI_laptop/ifdda_results_sorted.csv"

df = pd.read_csv(CSV_PATH)

# Sort: N -> OMP -> EXE (custom) -> rep
df_sorted = df.sort_values(by=["N", "OMP", "rep"])

df_sorted.to_csv(SORTED_CSV_PATH, index=False)
print(f"Sorted CSV written to: {SORTED_CSV_PATH}")
