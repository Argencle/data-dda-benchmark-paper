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

# ddscatcli environment variables
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
DDSCAT_LOG_DIR="$LOG_ROOT/ddscat"
mkdir -p "$ADDA_LOG_DIR" "$DDSCAT_LOG_DIR"

###############################################################################
# Main loop over N
###############################################################################

for N in 15 25; do
  echo
  echo "=============================="
  echo "          N = ${N}"
  echo "=============================="

  ###########################################################################
  # ADDA / MPI (grid = N)
  ###########################################################################
  echo
  echo "### ADDA (MPI) runs, N=${N}"
  cd "$ADDA_DIR"

  # Keep OpenMP threads at 1 to avoid oversubscription during MPI runs
  export OMP_NUM_THREADS=1

  ADDA_CMD="./adda_mpi -shape box 1 1 -size 2.3873241463784303 -lambda 0.5 -m 1.313 0.0 \
    -init_field zero -grid ${N} -eps 4.024 -iter bicgstab -pol fcd -int fcd -scat dr -ntheta 10"

  for NP in 1 2 5 10 15; do
    echo
    echo "---- ADDA: N=${N}, mpirun -np ${NP} ----"
    /usr/bin/time -v mpirun -np "$NP" $ADDA_CMD

    # After each ADDA run:
    # - find the run directory (ADDA_DIR/run*/),
    # - move the file run*/log to the ADDA_LOG_DIR,
    # - remove the run directory to avoid conflicts with the next runs.

    # Allow globbing with no matches to return an empty array instead of literal pattern
    shopt -s nullglob
    run_dirs=( "$ADDA_DIR"/run*/ )
    shopt -u nullglob

    if ((${#run_dirs[@]} == 0)); then
      echo "WARNING: no run* directory found in $ADDA_DIR after ADDA (N=${N}, NP=${NP})"
    else
      # Take the last directory in the list (usually the most recent: run0001, run0002, ...)
      idx=$((${#run_dirs[@]} - 1))
      latest_run="${run_dirs[$idx]}"
      latest_run="${latest_run%/}"  # remove trailing slash

      run_name="$(basename "$latest_run")"
      log_file="${latest_run}/log"  # file is always named 'run*/log'

      if [ -f "$log_file" ]; then
        dest_file="$ADDA_LOG_DIR/adda_N=${N}_np=${NP}.log"
        echo "Moving ADDA log file: $log_file -> $dest_file"
        mv "$log_file" "$dest_file"

        echo "Removing ADDA run directory: $latest_run"
        rm -rf "$latest_run"
      else
        echo "WARNING: expected log file not found: $log_file (N=${N}, NP=${NP})"
      fi
    fi
  done

  ###########################################################################
  # IFDDA / OpenMP (-nnnr = N)
  ###########################################################################
  echo
  echo "### IFDDA (OpenMP) runs, N=${N}"
  cd "$IFDDA_DIR"

  IFDDA_CMD="./ifdda -object cube 2387.3241463784303 -lambda 500 \
    -epsmulti 1.7239689999999999 0.0 \
    -ninitest 0 -nnnr ${N} -tolinit 9.46237161365793d-5 \
    -methodeit BICGSTAB -polarizability FG"

  for OMP in 1 2 5 10 15 22; do
    echo
    echo "---- IFDDA: N=${N}, OMP_NUM_THREADS=${OMP} ----"
    export OMP_NUM_THREADS="$OMP"
    /usr/bin/time -v $IFDDA_CMD
  done

  ###########################################################################
  # DDSCAT / OpenMP (MKL and GPFA) (MEM_ALLOW / SHPAR = N N N)
  ###########################################################################
  echo
  echo "### DDSCAT (OpenMP) runs, N=${N}"
  cd "$DDSCAT_DIR"

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

  DDSCAT_LOG_SRC="$DDSCAT_DIR/ddscat7.3.4_250505/ddscat.log_000"

  for FFT in FFTMKL GPFAFT; do
    for OMP in 1 2 5 10 15; do
      echo
      echo "---- DDSCAT: N=${N}, OMP_NUM_THREADS=${OMP}, FFT=${FFT} ----"
      export OMP_NUM_THREADS="$OMP"
      /usr/bin/time -v ddscatcli "${DDSCAT_ARGS_BASE[@]}" -CMDFFT "$FFT" -run

      if [ -f "$DDSCAT_LOG_SRC" ]; then
        dest_log="$DDSCAT_LOG_DIR/ddscat_fft_N=${N}-${FFT}_omp-${OMP}.log"
        echo "Moving DDSCAT log file: $DDSCAT_LOG_SRC -> $dest_log"
        mv "$DDSCAT_LOG_SRC" "$dest_log"
      else
        echo "WARNING: DDSCAT log not found at $DDSCAT_LOG_SRC (N=${N}, FFT=${FFT}, OMP=${OMP})"
      fi
    done
  done

done  # end loop over N

echo
echo "===== DDA batch run finished: $(date) ====="
