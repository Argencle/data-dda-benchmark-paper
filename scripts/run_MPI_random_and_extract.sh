#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Project setup
###############################################################################

PROJECT_ROOT="<Path_to_DDA_project_root>"  # <--- change this to your DDA project root directory

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

LOG_ROOT="$PROJECT_ROOT/logs_MPI_laptop"
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
  echo "N,NP,FFT,rep,first_residual,last_residual,Cext,Qext,elapsed_time,total_iterations,total_matvec,total_wall_time,solver_time,fft_time" > "$ADDA_CSV"
fi

if [ ! -f "$IFDDA_CSV" ]; then
  echo "N,OMP,EXE,rep,first_residual,last_residual,Cext,elapsed_time,num_iterations,num_matvec,total_time,solver_time" > "$IFDDA_CSV"
fi

# NOTE: DDSCAT now has an extra column elapsed_seconds
if [ ! -f "$DDSCAT_CSV" ]; then
  echo "N,FFT,OMP,rep,first_residual,last_residual,Qext,elapsed_time,elapsed_seconds,num_iterations" > "$DDSCAT_CSV"
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
# Build job list with repetitions
###############################################################################

REPEATS=3                 # <--- change this for more/less repetitions
N_VALUES=(150 250)        # <--- change N list if needed
ADDA_NP=(1 2 5 10 15)
ADDA_FFT=(FFTW GPFA)      # ADDA FFT modes (binaries: adda_mpi / adda_mpi_gpfa)
IFDDA_OMP=(1 2 5 10 15 22)
IFDDA_EXES=(ifdda)  # list of IFDDA executables
DDSCAT_FFT=(FFTMKL GPFAFT)
DDSCAT_OMP=(1 2 5 10 15 22)

jobs=()

for rep in $(seq 1 "$REPEATS"); do
  for N in "${N_VALUES[@]}"; do
    # ADDA jobs
    for NP in "${ADDA_NP[@]}"; do
      for FFT in "${ADDA_FFT[@]}"; do
        jobs+=("ADDA $N $NP $FFT $rep")
      done
    done

    # IFDDA jobs (one job per executable)
    for OMP in "${IFDDA_OMP[@]}"; do
      for EXE in "${IFDDA_EXES[@]}"; do
        jobs+=("IFDDA $N $OMP $EXE $rep")
      done
    done

    # DDSCAT jobs
    for FFT in "${DDSCAT_FFT[@]}"; do
      for OMP in "${DDSCAT_OMP[@]}"; do
        jobs+=("DDSCAT $N $FFT $OMP $rep")
      done
    done
  done
done

echo
echo "Number of jobs to run (including repetitions): ${#jobs[@]}"
echo "Randomizing job order..."
echo

# Shuffle into an array (avoid pipeline/subshell issues with set -e)
mapfile -t shuffled_jobs < <(printf '%s\n' "${jobs[@]}" | shuf)

###############################################################################
# Run all jobs in random order
###############################################################################

for line in "${shuffled_jobs[@]}"; do
  [ -z "$line" ] && continue

  set -- $line
  kind="$1"

  case "$kind" in
    #########################################################################
    # ADDA job: ADDA N NP FFT rep
    #########################################################################
    ADDA)
      N="$2"
      NP="$3"
      FFT="$4"
      REP="$5"

      echo
      echo "=============================="
      echo " ADDA job: N=${N}, NP=${NP}, FFT=${FFT}, rep=${REP}"
      echo "=============================="

      cd "$ADDA_DIR"
      export OMP_NUM_THREADS=1  # avoid oversubscription with MPI

      # Choose executable based on FFT mode
      case "$FFT" in
        FFTW)
          ADDA_EXE="./adda_mpi"
          ;;
        GPFA)
          ADDA_EXE="./adda_mpi_gpfa"
          ;;
        *)
          echo "Unknown ADDA FFT mode: $FFT" >&2
          exit 1
          ;;
      esac

      ADDA_CMD="$ADDA_EXE -shape box 1 1 -size 2387.3241463784303 -lambda 500 -m 1.313 0.0 \
        -init_field zero -grid ${N} -eps 4.024 -iter bicgstab -pol fcd -int fcd -scat dr -ntheta 10"

      ADDA_STDOUT="$ADDA_LOG_DIR/adda_stdout_N=${N}_np=${NP}_FFT=${FFT}_rep=${REP}.log"
      /usr/bin/time -v mpirun -np "$NP" $ADDA_CMD 2>&1 | tee "$ADDA_STDOUT"

      # Handle ADDA internal run directory: run*/log
      shopt -s nullglob
      run_dirs=( "$ADDA_DIR"/run*/ )
      shopt -u nullglob

      ADDA_RUN_LOG=""

      if ((${#run_dirs[@]} == 0)); then
        echo "WARNING: no run* directory found in $ADDA_DIR after ADDA (N=${N}, NP=${NP}, FFT=${FFT}, rep=${REP})"
      else
        idx=$((${#run_dirs[@]} - 1))
        latest_run="${run_dirs[$idx]%/}"  # remove trailing slash
        log_file="${latest_run}/log"

        if [ -f "$log_file" ]; then
          ADDA_RUN_LOG="$ADDA_LOG_DIR/adda_log_N=${N}_np=${NP}_FFT=${FFT}_rep=${REP}.log"
          echo "Moving ADDA log file: $log_file -> $ADDA_RUN_LOG"
          mv "$log_file" "$ADDA_RUN_LOG"

          echo "Removing ADDA run directory: $latest_run"
          rm -rf "$latest_run"
        else
          echo "WARNING: expected log file not found: $log_file (N=${N}, NP=${NP}, FFT=${FFT}, rep=${REP})"
        fi
      fi

      # ---- Extract metrics from ADDA stdout and run log ----
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

      echo "${N},${NP},${FFT},${REP},${first_residual},${last_residual},${cext_val},${qext_val},${elapsed_time},${total_iterations},${total_matvec},${total_wall_time},${solver_time},${fft_time}" >> "$ADDA_CSV"
      ;;

    #########################################################################
    # IFDDA job: IFDDA N OMP EXE rep
    #########################################################################
    IFDDA)
      N="$2"
      OMP="$3"
      EXE="$4"
      REP="$5"

      echo
      echo "=============================="
      echo " IFDDA job: N=${N}, OMP=${OMP}, EXE=${EXE}, rep=${REP}"
      echo "=============================="

      cd "$IFDDA_DIR"
      export OMP_NUM_THREADS="$OMP"

      IFDDA_CMD="./${EXE} -object cube 2387.3241463784303 -lambda 500 \
        -epsmulti 1.723969 0.0 \
        -ninitest 0 -nnnr ${N} -tolinit 9.46237161365793d-5 \
        -methodeit BICGSTAB -polarizability FG"

      IFDDA_STDOUT="$IFDDA_LOG_DIR/ifdda_stdout_${EXE}_N=${N}_omp=${OMP}_rep=${REP}.log"
      /usr/bin/time -v $IFDDA_CMD 2>&1 | tee "$IFDDA_STDOUT"

      res_lines=$(grep 'RESIDU' "$IFDDA_STDOUT" 2>/dev/null | grep -v 'INIT' || true)
      first_res_line=$(echo "$res_lines" | head -n1 || true)
      last_res_line=$(echo "$res_lines"  | tail -n1 || true)

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

      first_residual=$(safe_value "$first_residual")
      last_residual=$(safe_value "$last_residual")
      cext_val=$(safe_value "$cext_val")
      elapsed_time=$(safe_value "$elapsed_time")
      num_iterations=$(safe_value "$num_iterations")
      num_matvec=$(safe_value "$num_matvec")
      total_time=$(safe_value "$total_time")
      solver_time=$(safe_value "$solver_time")

      echo "${N},${OMP},${EXE},${REP},${first_residual},${last_residual},${cext_val},${elapsed_time},${num_iterations},${num_matvec},${total_time},${solver_time}" >> "$IFDDA_CSV"
      ;;

    #########################################################################
    # DDSCAT job: DDSCAT N FFT OMP rep
    #########################################################################
    DDSCAT)
      N="$2"
      FFT="$3"
      OMP="$4"
      REP="$5"

      echo
      echo "=============================="
      echo " DDSCAT job: N=${N}, FFT=${FFT}, OMP=${OMP}, rep=${REP}"
      echo "=============================="

      cd "$DDSCAT_DIR"
      DDSCAT_LOG_SRC="$DDSCAT_DIR/ddscat7.3.4_250505/ddscat.log_000"

      DDSCAT_ARGS_BASE=(
        -CSHAPE RCTGLPRSM
        -AEFF "1480.9777061418503 1480.9777061418503 1 'LIN'"
        -WAVELENGTHS "500 500 1 'LIN'"
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

      export OMP_NUM_THREADS="$OMP"

      DDSCAT_STDOUT="$DDSCAT_LOG_DIR/ddscat_stdout_N=${N}_FFT=${FFT}_omp=${OMP}_rep=${REP}.log"
      /usr/bin/time -v ddscatcli "${DDSCAT_ARGS_BASE[@]}" -CMDFFT "$FFT" -run 2>&1 | tee "$DDSCAT_STDOUT"

      DDSCAT_RUN_LOG=""
      if [ -f "$DDSCAT_LOG_SRC" ]; then
        DDSCAT_RUN_LOG="$DDSCAT_LOG_DIR/ddscat_fft_N=${N}-${FFT}_omp-${OMP}_rep=${REP}.log"
        echo "Moving DDSCAT log file: $DDSCAT_LOG_SRC -> $DDSCAT_RUN_LOG"
        mv "$DDSCAT_LOG_SRC" "$DDSCAT_RUN_LOG"
      else
        echo "WARNING: DDSCAT log not found at $DDSCAT_LOG_SRC (N=${N}, FFT=${FFT}, OMP=${OMP}, rep=${REP})"
      fi

      # Elapsed time string (m:ss or h:mm:ss)
      elapsed_line=$(grep 'Elapsed (wall clock) time (h:mm:ss or m:ss):' "$DDSCAT_STDOUT" 2>/dev/null | tail -n1 || true)
      elapsed_time=$(echo "$elapsed_line" | sed -E 's/.*Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):[[:space:]]*//' || true)

      # Convert elapsed_time to seconds as float (elapsed_seconds)
      elapsed_seconds=""
      if [ -n "$elapsed_time" ]; then
        # handle "m:ss.xx" or "h:mm:ss.xx"
        elapsed_seconds=$(echo "$elapsed_time" | awk -F: '
          {
            if (NF==2) {
              m=$1; s=$2;
              printf "%.4f", (m*60)+s;
            } else if (NF==3) {
              h=$1; m=$2; s=$3;
              printf "%.4f", (h*3600)+(m*60)+s;
            }
          }' || true)
      fi

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

      elapsed_time=$(safe_value "$elapsed_time")
      elapsed_seconds=$(safe_value "$elapsed_seconds")
      first_residual=$(safe_value "$first_residual")
      last_residual=$(safe_value "$last_residual")
      qext_val=$(safe_value "$qext_val")
      num_iterations=$(safe_value "$num_iterations")

      echo "${N},${FFT},${OMP},${REP},${first_residual},${last_residual},${qext_val},${elapsed_time},${elapsed_seconds},${num_iterations}" >> "$DDSCAT_CSV"
      ;;

    *)
      echo "Unknown job kind: $kind" >&2
      exit 1
      ;;
  esac

done

echo
echo "===== DDA batch run finished: $(date) ====="
