import os
import re
import glob
import pandas as pd


HMS_RE = re.compile(r"\b(\d{2}):(\d{2}):(\d{2})\b")


def hms_to_seconds(hms: str) -> int:
    """
    Convert a time string in HH:MM:SS format to total seconds.
    """
    m = HMS_RE.search(str(hms).strip())
    hh, mm, ss = map(int, m.groups())
    return hh * 3600 + mm * 60 + ss


def find_out_file_for_jobid(
    base_dir: str, jobid: int, logs_subdir: str = "slurm_logs"
) -> str:
    """
    Find the SLURM output file corresponding to a given JOBID.

    It searches inside:
        <base_dir>/<logs_subdir>/*_{JOBID}.out

    Example:
        logs_MPI_cluster/slurm_logs/mpi_ADDA_N150_np1_FFTFFTW_rep1_1595389.out
    """
    search_dir = os.path.join(base_dir, logs_subdir)
    pattern = os.path.join(search_dir, f"*_{jobid}.out")

    matches = glob.glob(pattern)

    # If multiple matches exist, pick the most recently modified file
    matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return matches[0]


def extract_elapsed_from_out(out_path: str, jobid: int) -> str:
    """
    Extract the Elapsed time (HH:MM:SS) from the SLURM output file.

    Strategy:
    - Look for all lines matching: ^\\s*{JOBID}.<step>
      Example: "1595389.0   adda_mpi ... 00:02:48"
    - Keep the last matching line (usually the ".0" step of the job)
    - Extract the last HH:MM:SS token from that line
    """
    line_re = re.compile(rf"^\s*{jobid}")
    last_candidate = None

    with open(out_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.rstrip("\n")
            if line_re.match(s):
                last_candidate = s

    # Extract the last HH:MM:SS occurrence on that line
    times = HMS_RE.findall(last_candidate)

    hh, mm, ss = times[0]
    return f"{hh}:{mm}:{ss}"


def fill_elapsed_in_csv(
    csv_path: str, out_csv_path: str, add_elapsed_seconds: bool
) -> None:
    """
    Fill missing elapsed_time values in a CSV by parsing the corresponding SLURM *.out files.

    - elapsed_time is filled with the HH:MM:SS value extracted from the job's *.out file
    - if add_elapsed_seconds is True (DDSCAT case), elapsed_seconds is also filled
    """
    df = pd.read_csv(csv_path)

    base_dir = os.path.dirname(os.path.abspath(csv_path))

    for idx, row in df.iterrows():

        jobid = int(row["JOBID"])

        out_path = find_out_file_for_jobid(
            base_dir, jobid, logs_subdir="slurm_logs"
        )

        elapsed = extract_elapsed_from_out(out_path, jobid)
        df.at[idx, "elapsed_time"] = elapsed

        if add_elapsed_seconds:
            df.at[idx, "elapsed_seconds"] = hms_to_seconds(elapsed)

    df.to_csv(out_csv_path, index=False)

    print(f"[OK] Wrote: {out_csv_path}")


def main():
    JOBS = [
        (
            "logs_MPI_cluster/adda_results_sorted.csv",
            "logs_MPI_cluster/adda_results_filled.csv",
            False,
        ),
        (
            "logs_MPI_cluster/ddscat_results_sorted.csv",
            "logs_MPI_cluster/ddscat_results_filled.csv",
            True,
        ),
        (
            "logs_MPI_cluster/ifdda_results_sorted.csv",
            "logs_MPI_cluster/ifdda_results_filled.csv",
            False,
        ),
    ]

    for in_csv, out_csv, need_secs in JOBS:
        fill_elapsed_in_csv(in_csv, out_csv, add_elapsed_seconds=need_secs)


if __name__ == "__main__":
    main()
