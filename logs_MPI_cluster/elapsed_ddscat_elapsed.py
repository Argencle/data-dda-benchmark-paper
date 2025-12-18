import re
import sys
from pathlib import Path

import pandas as pd


def hhmmss_to_seconds(s: str) -> int:
    h, m, sec = s.split(":")
    return int(h) * 3600 + int(m) * 60 + int(sec)


def extract_ddscatcli_elapsed(log_path: Path, jobid: int) -> str | None:
    """
    Extrait 'HH:MM:SS' depuis la table sacct du .out, en ciblant la ligne ddscatcli (ou JobID=<jobid>.0).
    """
    if not log_path.exists():
        return None

    text = log_path.read_text(errors="replace")

    # 1) Repérer la table sacct qui contient la colonne Elapsed (celle avec JobName, Start, End, Elapsed, ...)
    # On prend le bloc après la ligne "$ sacct --format='JobID,Partition,JobName,...,Elapsed,...'"
    m = re.search(
        r"^\$ sacct --format=.*?Elapsed.*?$([\s\S]*?)(?:^\$ sacct|\Z)",
        text,
        flags=re.MULTILINE,
    )
    if not m:
        return None

    block = m.group(1)

    # 2) Dans ce bloc, chercher la ligne ddscatcli (priorité), sinon JobID "<jobid>.0"
    # Les colonnes sacct sont séparées par des espaces multiples -> on split sur 2+ espaces.
    lines = [ln.rstrip() for ln in block.splitlines() if ln.strip()]

    # Filtrer les lignes de données (ignorer en-têtes "JobID Partition JobName ..." et séparateurs "-----")
    data_lines = []
    for ln in lines:
        if ln.lstrip().startswith("JobID") or set(ln.strip()) == {"-"}:
            continue
        data_lines.append(ln)

    if not data_lines:
        return None

    def parse_elapsed_from_line(ln: str) -> str | None:
        # split en colonnes sur 2+ espaces (robuste vs colonnes vides)
        cols = re.split(r"\s{2,}", ln.strip())
        # On sait juste qu'il y a un champ "Elapsed" qui ressemble à HH:MM:SS,
        # donc on le prend par motif, mais *sur une ligne ddscatcli*.
        for c in cols:
            if re.fullmatch(r"\d{2}:\d{2}:\d{2}", c):
                return c
        return None

    # Priorité 1: ligne contenant ddscatcli
    for ln in data_lines:
        if re.search(r"\bddscatcli\b", ln):
            return parse_elapsed_from_line(ln)

    # Priorité 2: JobID exact "<jobid>.0"
    jid0 = f"{jobid}.0"
    for ln in data_lines:
        if ln.strip().startswith(jid0):
            return parse_elapsed_from_line(ln)

    return None


def main(csv_in: Path, csv_out: Path, logs_dir: Path) -> int:
    df = pd.read_csv(csv_in, na_values=["NA", "NaN", "nan", ""])

    for col in ["JOBID", "N", "OMP", "rep"]:
        df[col] = df[col].astype(int)
    df["FFT"] = df["FFT"].astype(str)

    mask = df["elapsed_time"].isna() | df["elapsed_seconds"].isna()

    missing_logs = []
    missing_elapsed = []

    for idx, row in df[mask].iterrows():
        jobid = int(row["JOBID"])
        n = int(row["N"])
        fft = str(row["FFT"])
        omp = int(row["OMP"])
        rep = int(row["rep"])

        log_name = f"mpi_DDSCAT_N{n}_FFT{fft}_omp{omp}_rep{rep}_{jobid}.out"
        log_path = logs_dir / log_name

        elapsed = extract_ddscatcli_elapsed(log_path, jobid=jobid)
        if elapsed is None:
            if not log_path.exists():
                missing_logs.append(log_name)
            else:
                missing_elapsed.append(log_name)
            continue

        df.at[idx, "elapsed_time"] = elapsed
        df.at[idx, "elapsed_seconds"] = hhmmss_to_seconds(elapsed)

    df.to_csv(csv_out, index=False)

    if missing_logs:
        print(f"[WARN] {len(missing_logs)} logs manquants (exemples):")
        for n in missing_logs[:10]:
            print("  -", n)
    if missing_elapsed:
        print(
            f"[WARN] {len(missing_elapsed)} logs trouvés mais elapsed ddscatcli non détecté (exemples):"
        )
        for n in missing_elapsed[:10]:
            print("  -", n)

    print(f"OK: écrit -> {csv_out}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python fill_elapsed.py input.csv output.csv slurm_logs/")
        sys.exit(2)

    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])))
