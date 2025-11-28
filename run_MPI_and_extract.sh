#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Project setup
###############################################################################

PROJECT_ROOT="/home/argentic@coria.fr/Bureau/Work/paper1"

VENV_DIR="$PROJECT_ROOT/.venv"
ADDA_DIR="$PROJECT_ROOT/adda/src/mpi"
IFDDA_DIR="$PROJECT_ROOT/if-dda/tests/test_command"
DDSCAT_DIR="$PROJECT_ROOT"

# DDSCAT environment variables
export DDSCAT_PAR="$PROJECT_ROOT/ddscat7.3.4_250505/ddscat.par"
export DDSCAT_EXE="$PROJECT_ROOT/ddscat7.3.4_250505/src/ddscat"

# Activate Python virtual environment
source "$VENV_DIR/bin/activate"

###############################################################################
# Logging setup (global script log)
###############################################################################

LOG_ROOT="$PROJECT_ROOT/logs"
mkdir -p "$LOG_ROOT"
LOG="$LOG_ROOT/run_$(date +'%Y%m%d_%H%M%S').log"

# Send both stdout and stderr to the log (and still echo to terminal)
exec > >(tee -a "$LOG") 2>&1

echo "===== DDA batch run started: $(date) on $(hostname) ====="
echo "Main log file: $LOG"
echo

# Directories for simulation logs
ADDA_LOG_DIR="$LOG_ROOT/adda"
IFDDA_LOG_DIR="$LOG_ROOT/ifdda"
DDSCAT_LOG_DIR="$LOG_ROOT/ddscat"
mkdir -p "$ADDA_LOG_DIR" "$IFDDA_LOG_DIR" "$DDSCAT_LOG_DIR"

# CSV result files
ADDA_CSV="$LOG_ROOT/adda_results.csv"
IFDDA_CSV="$LOG_ROOT/ifdda_results.csv"
DDSCAT_CSV="$LOG_ROOT/ddscat_results.csv"

# Create CSV headers if they do not exist
if [ ! -f "$ADDA_CSV" ]; then
  echo "N,NP,first_residual,last_residual,Cext,Qext,elapsed_time,total_iterations,total_matvec,total_wall_time,solver_time,fft_time" > "$ADDA_CSV"
fi

if [ ! -f "$IFDDA_CSV" ]; then
  echo "N,OMP,first_residual,last_residual,Cext,elapsed_time,num_iterations,num_matvec,total_time,solver_time" > "$IFDDA_CSV"
fi

if [ ! -f "$DDSCAT_CSV" ]; then
  echo "N,FFT,OMP,first_residual,last_residual,Qext,elapsed_time,num_iterations" > "$DDSCAT_CSV"
fi

###############################################################################
# Helper function: safe value (default NA on empty)
###############################################################################
safe_value() {
  local v="$1"
  if [ -z "$v" ]; then
    echo "NA"
  else
    echo "$v"
  fi
}

###############################################################################
# Main loop over N
###############################################################################

for N in 150; do
  echo
  echo "=============================="
  echo "          N = ${N}"
  echo "=============================="

  ###########################################################################
  # Build simulation commands for this N
  ###########################################################################
  ADDA_CMD="./adda_mpi -shape box 1 1 -size 2.3873241463784303 -lambda 0.5 -m 1.313 0.0 \
    -init_field zero -grid ${N} -eps 4.024 -iter bicgstab -pol fcd -int fcd -scat dr -ntheta 10"

  IFDDA_CMD="./ifdda -object cube 2387.3241463784303 -lambda 500 \
    -epsmulti 1.7239689999999999 0.0 \
    -ninitest 0 -nnnr ${N} -tolinit 9.46237161365793d-5 \
    -methodeit BICGSTAB -polarizability FG"

  DDSCAT_ARGS_BASE=(
    -CSHAPE RCTGLPRSM
    -AEFF "1.4809777061418503 1.4809777061418503 1 'LIN'"
    -WAVELENGTHS "0.5 0.5 1 'LIN'"
    -DIEL 'diel/m1.313_0.00'
    -MEM_ALLOW "${N} ${N} ${N}"
    -SHPAR "${N} ${N} ${N}"
    -TOL 9.46237161365793e-5
    -CMDSOL PBCGST
    -CALPHA FLTRCD
    -ETASCA 10
    -IORTH 1
    -NPLANES 0
    -NRFLD 0
  )

  ###########################################################################
  # ADDA / MPI (grid = N)
  ###########################################################################
  echo
  echo "### ADDA (MPI) runs, N=${N}"
  cd "$ADDA_DIR"

  # Keep OpenMP threads at 1 to avoid oversubscription during MPI runs
  export OMP_NUM_THREADS=1

  for NP in 1 2; do
    echo
    echo "---- ADDA: N=${N}, mpirun -np ${NP} ----"

    # Per-run stdout log
    ADDA_STDOUT="$ADDA_LOG_DIR/adda_stdout_N=${N}_np=${NP}.log"
    /usr/bin/time -v mpirun -np "$NP" $ADDA_CMD 2>&1 | tee "$ADDA_STDOUT"

    # After each ADDA run:
    # - find the run directory (ADDA_DIR/run*/),
    # - move the file run*/log to ADDA_LOG_DIR,
    # - remove the run directory to avoid conflicts with next runs.

    shopt -s nullglob
    run_dirs=( "$ADDA_DIR"/run*/ )
    shopt -u nullglob

    ADDA_RUN_LOG=""

    if ((${#run_dirs[@]} == 0)); then
      echo "WARNING: no run* directory found in $ADDA_DIR after ADDA (N=${N}, NP=${NP})"
    else
      # Take the last directory in the list (usually the most recent)
      idx=$((${#run_dirs[@]} - 1))
      latest_run="${run_dirs[$idx]}"
      latest_run="${latest_run%/}"  # remove trailing slash

      log_file="${latest_run}/log"  # file is always named 'run*/log'

      if [ -f "$log_file" ]; then
        ADDA_RUN_LOG="$ADDA_LOG_DIR/adda_log_N=${N}_np=${NP}.log"
        echo "Moving ADDA log file: $log_file -> $ADDA_RUN_LOG"
        mv "$log_file" "$ADDA_RUN_LOG"

        echo "Removing ADDA run directory: $latest_run"
        rm -rf "$latest_run"
      else
        echo "WARNING: expected log file not found: $log_file (N=${N}, NP=${NP})"
      fi
    fi

    #######################################################################
    # Extract metrics from ADDA stdout and run log
    #######################################################################

    # From stdout:
    # - first residual: first RE_001 = ...
    # - last residual: last RE_... = ...
    # - Cext = ...
    # - Qext = ...
    # - elapsed time from /usr/bin/time (string after full label)

    first_re_line=$(grep 'RE_001' "$ADDA_STDOUT" 2>/dev/null | head -n1 || true)
    last_re_line=$(grep 'RE_[0-9][0-9][0-9]' "$ADDA_STDOUT" 2>/dev/null | tail -n1 || true)

    first_residual=$(echo "$first_re_line" | awk -F'= *' '{print $2}' | awk '{print $1}' || true)
    last_residual=$(echo "$last_re_line"  | awk -F'= *' '{print $2}' | awk '{print $1}' || true)

    cext_line=$(grep 'Cext' "$ADDA_STDOUT" 2>/dev/null | head -n1 || true)
    qext_line=$(grep 'Qext' "$ADDA_STDOUT" 2>/dev/null | head -n1 || true)

    cext_val=$(echo "$cext_line" | awk -F'= *' '{print $2}' | awk '{print $1}' || true)
    qext_val=$(echo "$qext_line" | awk -F'= *' '{print $2}' | awk '{print $1}' || true)

    elapsed_line=$(grep 'Elapsed (wall clock) time (h:mm:ss or m:ss):' "$ADDA_STDOUT" 2>/dev/null | tail -n1 || true)
    elapsed_time=$(echo "$elapsed_line" | sed -E 's/.*Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):[[:space:]]*//' || true)

    # From ADDA log file:
    # - Total number of iterations:
    # - Total number of matrix-vector products:
    # - Total wall time:
    # - Internal fields: (solver time)
    # - matvec products: (FFT time)

    total_iterations=""
    total_matvec=""
    total_wall_time=""
    solver_time=""
    fft_time=""

    if [ -n "${ADDA_RUN_LOG:-}" ] && [ -f "$ADDA_RUN_LOG" ]; then
      it_line=$(grep 'Total number of iterations' "$ADDA_RUN_LOG" 2>/dev/null | head -n1 || true)
      mv_line=$(grep 'Total number of matrix-vector products' "$ADDA_RUN_LOG" 2>/dev/null | head -n1 || true)
      tw_line=$(grep 'Total wall time' "$ADDA_RUN_LOG" 2>/dev/null | head -n1 || true)
      sol_line=$(grep 'Internal fields' "$ADDA_RUN_LOG" 2>/dev/null | head -n1 || true)
      fft_line=$(grep 'matvec products' "$ADDA_RUN_LOG" 2>/dev/null | head -n1 || true)

      total_iterations=$(echo "$it_line" | awk -F': *' '{print $2}' | awk '{print $1}' || true)
      total_matvec=$(echo "$mv_line"      | awk -F': *' '{print $2}' | awk '{print $1}' || true)
      total_wall_time=$(echo "$tw_line"   | awk -F': *' '{print $2}' || true)
      solver_time=$(echo "$sol_line"      | awk -F': *' '{print $2}' || true)
      fft_time=$(echo "$fft_line"         | awk -F': *' '{print $2}' || true)
    fi

    # Fallback NA for missing values
    first_residual=$(safe_value "$first_residual")
    last_residual=$(safe_value "$last_residual")
    cext_val=$(safe_value "$cext_val")
    qext_val=$(safe_value "$qext_val")
    elapsed_time=$(safe_value "$elapsed_time")
    total_iterations=$(safe_value "$total_iterations")
    total_matvec=$(safe_value "$total_matvec")
    total_wall_time=$(safe_value "$total_wall_time")
    solver_time=$(safe_value "$solver_time")
    fft_time=$(safe_value "$fft_time")

    # Append to ADDA CSV
    echo "${N},${NP},${first_residual},${last_residual},${cext_val},${qext_val},${elapsed_time},${total_iterations},${total_matvec},${total_wall_time},${solver_time},${fft_time}" >> "$ADDA_CSV"

  done  # NP

  ###########################################################################
  # IFDDA / OpenMP (-nnnr = N)
  ###########################################################################
  echo
  echo "### IFDDA (OpenMP) runs, N=${N}"
  cd "$IFDDA_DIR"

  for OMP in 1 2; do
    echo
    echo "---- IFDDA: N=${N}, OMP_NUM_THREADS=${OMP} ----"
    export OMP_NUM_THREADS="$OMP"

    IFDDA_STDOUT="$IFDDA_LOG_DIR/ifdda_stdout_N=${N}_omp=${OMP}.log"
    /usr/bin/time -v $IFDDA_CMD 2>&1 | tee "$IFDDA_STDOUT"

    # From stdout:
    # - first residual (first RESIDU, ignoring INIT RESIDU)
    # - last residual (last RESIDU, ignoring INIT RESIDU)
    # - Cext =
    # - elapsed time from /usr/bin/time
    # - Number of iterations              :
    # - Number of product Ax needs        :
    # - Real time to execute the code in second :
    # - Real time to solve Ax=b in second :

    # Get all RESIDU lines except the INIT RESIDU
    res_lines=$(grep 'RESIDU' "$IFDDA_STDOUT" 2>/dev/null | grep -v 'INIT' || true)
    first_res_line=$(echo "$res_lines" | head -n1 || true)
    last_res_line=$(echo "$res_lines"  | tail -n1 || true)

    # Residual lines look like: " RESIDU   1.2345E-003"
    # -> take the last field on the line
    first_residual=$(echo "$first_res_line" | awk '{print $NF}' || true)
    last_residual=$(echo "$last_res_line"  | awk '{print $NF}' || true)


    cext_line=$(grep 'Cext' "$IFDDA_STDOUT" 2>/dev/null | head -n1 || true)
    cext_val=$(echo "$cext_line" | awk -F'= *' '{print $2}' | awk '{print $1}' || true)

    elapsed_line=$(grep 'Elapsed (wall clock) time (h:mm:ss or m:ss):' "$IFDDA_STDOUT" 2>/dev/null | tail -n1 || true)
    elapsed_time=$(echo "$elapsed_line" | sed -E 's/.*Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):[[:space:]]*//' || true)

    it_line=$(grep 'Number of iterations' "$IFDDA_STDOUT" 2>/dev/null | head -n1 || true)
    matvec_line=$(grep 'Number of product Ax needs' "$IFDDA_STDOUT" 2>/dev/null | head -n1 || true)
    tot_time_line=$(grep 'Real time to execute the code in second' "$IFDDA_STDOUT" 2>/dev/null | head -n1 || true)
    sol_time_line=$(grep 'Real time to solve Ax=b in second' "$IFDDA_STDOUT" 2>/dev/null | head -n1 || true)

    num_iterations=$(echo "$it_line"       | awk -F': *' '{print $2}' | awk '{print $1}' || true)
    num_matvec=$(echo "$matvec_line"       | awk -F': *' '{print $2}' | awk '{print $1}' || true)
    total_time=$(echo "$tot_time_line"     | awk -F': *' '{print $2}' || true)
    solver_time=$(echo "$sol_time_line"    | awk -F': *' '{print $2}' || true)

    # Fallback NA
    first_residual=$(safe_value "$first_residual")
    last_residual=$(safe_value "$last_residual")
    cext_val=$(safe_value "$cext_val")
    elapsed_time=$(safe_value "$elapsed_time")
    num_iterations=$(safe_value "$num_iterations")
    num_matvec=$(safe_value "$num_matvec")
    total_time=$(safe_value "$total_time")
    solver_time=$(safe_value "$solver_time")

    # Append to IFDDA CSV
    echo "${N},${OMP},${first_residual},${last_residual},${cext_val},${elapsed_time},${num_iterations},${num_matvec},${total_time},${solver_time}" >> "$IFDDA_CSV"

  done  # OMP

  ###########################################################################
  # DDSCAT / OpenMP (MKL and GPFA) (MEM_ALLOW / SHPAR = N N N)
  ###########################################################################
  echo
  echo "### DDSCAT (OpenMP) runs, N=${N}"
  cd "$DDSCAT_DIR"

  DDSCAT_LOG_SRC="$DDSCAT_DIR/ddscat7.3.4_250505/ddscat.log_000"

  for FFT in FFTMKL GPFAFT; do
    for OMP in 1 2; do
      echo
      echo "---- DDSCAT: N=${N}, OMP_NUM_THREADS=${OMP}, FFT=${FFT} ----"
      export OMP_NUM_THREADS="$OMP"

      DDSCAT_STDOUT="$DDSCAT_LOG_DIR/ddscat_stdout_N=${N}_FFT=${FFT}_omp=${OMP}.log"
      /usr/bin/time -v ddscatcli "${DDSCAT_ARGS_BASE[@]}" -CMDFFT "$FFT" -run 2>&1 | tee "$DDSCAT_STDOUT"

      # Move DDSCAT internal log
      DDSCAT_RUN_LOG=""
      if [ -f "$DDSCAT_LOG_SRC" ]; then
        DDSCAT_RUN_LOG="$DDSCAT_LOG_DIR/ddscat_fft_N=${N}-${FFT}_omp-${OMP}.log"
        echo "Moving DDSCAT log file: $DDSCAT_LOG_SRC -> $DDSCAT_RUN_LOG"
        mv "$DDSCAT_LOG_SRC" "$DDSCAT_RUN_LOG"
      else
        echo "WARNING: DDSCAT log not found at $DDSCAT_LOG_SRC (N=${N}, FFT=${FFT}, OMP=${OMP})"
      fi

      #######################################################################
      # Extract metrics for DDSCAT
      #######################################################################

      # From stdout:
      # - elapsed time from /usr/bin/time

      elapsed_line=$(grep 'Elapsed (wall clock) time (h:mm:ss or m:ss):' "$DDSCAT_STDOUT" 2>/dev/null | tail -n1 || true)
      elapsed_time=$(echo "$elapsed_line" | sed -E 's/.*Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):[[:space:]]*//' || true)

      # From DDSCAT log file:
      # - first residual (first frac.err=)
      # - last residual (last frac.err=)
      # - extinction efficiency Q_ext=
      # - number of iterations = last iter=

      first_frac_line=""
      last_frac_line=""
      qext_line=""
      last_iter_line=""
      first_residual=""
      last_residual=""
      qext_val=""
      num_iterations=""

      if [ -n "${DDSCAT_RUN_LOG:-}" ] && [ -f "$DDSCAT_RUN_LOG" ]; then
        first_frac_line=$(grep 'frac.err=' "$DDSCAT_RUN_LOG" 2>/dev/null | head -n1 || true)
        last_frac_line=$(grep 'frac.err=' "$DDSCAT_RUN_LOG" 2>/dev/null | tail -n1 || true)
        qext_line=$(grep 'Q_ext=' "$DDSCAT_RUN_LOG" 2>/dev/null | head -n1 || true)
        last_iter_line=$(grep 'iter=' "$DDSCAT_RUN_LOG" 2>/dev/null | tail -n1 || true)

        first_residual=$(echo "$first_frac_line" | awk -F'frac.err=' '{print $2}' | awk '{print $1}' || true)
        last_residual=$(echo "$last_frac_line"  | awk -F'frac.err=' '{print $2}' | awk '{print $1}' || true)
        qext_val=$(echo "$qext_line"            | awk -F'Q_ext='   '{print $2}' | awk '{print $1}' || true)
        num_iterations=$(echo "$last_iter_line" | awk -F'iter='    '{print $2}' | awk '{print $1}' || true)
      fi

      # Fallback NA
      elapsed_time=$(safe_value "$elapsed_time")
      first_residual=$(safe_value "$first_residual")
      last_residual=$(safe_value "$last_residual")
      qext_val=$(safe_value "$qext_val")
      num_iterations=$(safe_value "$num_iterations")

      # Append to DDSCAT CSV
      echo "${N},${FFT},${OMP},${first_residual},${last_residual},${qext_val},${elapsed_time},${num_iterations}" >> "$DDSCAT_CSV"

    done  # OMP
  done    # FFT

done  # N

echo
echo "===== DDA batch run finished: $(date) ====="
