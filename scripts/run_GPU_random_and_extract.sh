#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Project setup
###############################################################################

PROJECT_ROOT="<Path_to_DDA_project_root>"  # <--- change this to your DDA project root directory

ADDA_DIR="$PROJECT_ROOT/adda/src/ocl"
IFDDA_DIR="$PROJECT_ROOT/if-dda/tests/test_command"

###############################################################################
# Logging setup (global script log) - GPU
###############################################################################

LOG_ROOT="$PROJECT_ROOT/logs_GPU_2000Ada"
mkdir -p "$LOG_ROOT"
LOG="$LOG_ROOT/run_GPU_$(date +'%Y%m%d_%H%M%S').log"

# Send both stdout and stderr to the log (and still echo to terminal)
exec > >(tee -a "$LOG") 2>&1

echo "===== DDA GPU batch run started: $(date) on $(hostname) ====="
echo "Main GPU log file: $LOG"
echo

# Directories for simulation logs
ADDA_LOG_DIR="$LOG_ROOT/adda"
IFDDA_LOG_DIR="$LOG_ROOT/ifdda"
mkdir -p "$ADDA_LOG_DIR" "$IFDDA_LOG_DIR"

# CSV result files (GPU)
ADDA_CSV="$LOG_ROOT/adda_gpu_results.csv"
IFDDA_CSV="$LOG_ROOT/ifdda_gpu_results.csv"

# Create CSV headers if they do not exist
if [ ! -f "$ADDA_CSV" ]; then
  echo "N,exe,solver,rep,first_residual,last_residual,Cext,Qext,elapsed_time,total_iterations,total_matvec,total_wall_time,solver_time,fft_time" > "$ADDA_CSV"
fi

if [ ! -f "$IFDDA_CSV" ]; then
  echo "N,exe,OMP,rep,first_residual,last_residual,Cext,elapsed_time,num_iterations,num_matvec,total_time,solver_time" > "$IFDDA_CSV"
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
# Build GPU job list with repetitions + randomization
###############################################################################

REPEATS=3                        # <--- change this to run each job X times
ADDA_N_VALUES=(150 250)
ADDA_EXES=(adda_ocl adda_ocl_blas)
IFDDA_N=150                      # IFDDA GPU only for N=150
IFDDA_EXES=(ifdda_gpu_sp ifdda_gpu)
IFDDA_OMP_VALUES=(1 10)

jobs=()

for rep in $(seq 1 "$REPEATS"); do
  # ADDA GPU jobs
  for N in "${ADDA_N_VALUES[@]}"; do
    for exe_name in "${ADDA_EXES[@]}"; do
      if [ "$exe_name" = "adda_ocl_blas" ] && [ "$N" -ne 150 ]; then
        continue
      fi
      if [ "$exe_name" = "adda_ocl" ]; then
        solvers=(bicgstab bicg)
      else
        solvers=(bicg)
      fi
      for solver in "${solvers[@]}"; do
        jobs+=("ADDA $N $exe_name $solver $rep")
      done
    done
  done

  # IFDDA GPU jobs (only N=150)
  N="$IFDDA_N"
  for exe_name in "${IFDDA_EXES[@]}"; do
    for OMP in "${IFDDA_OMP_VALUES[@]}"; do
      jobs+=("IFDDA $N $exe_name $OMP $rep")
    done
  done
done

echo
echo "Number of GPU jobs to run (including repetitions): ${#jobs[@]}"
echo "Randomizing GPU job order..."
echo

# Shuffle jobs (no subshell issue with set -e)
mapfile -t shuffled_jobs < <(printf '%s\n' "${jobs[@]}" | shuf)

###############################################################################
# Run all GPU jobs in random order
###############################################################################

for line in "${shuffled_jobs[@]}"; do
  [ -z "$line" ] && continue

  set -- $line
  kind="$1"

  case "$kind" in
    #########################################################################
    # ADDA GPU job: ADDA N exe_name solver rep
    #########################################################################
    ADDA)
      N="$2"
      exe_name="$3"
      solver="$4"
      REP="$5"

      echo
      echo "=============================="
      echo " ADDA GPU job: N=${N}, exe=${exe_name}, solver=${solver}, rep=${REP}"
      echo "=============================="

      cd "$ADDA_DIR"
      export OMP_NUM_THREADS=1

      exe="./${exe_name}"
      if [ ! -x "$exe" ]; then
        echo "WARNING: executable $exe not found or not executable, skipping."
        continue
      fi

      ADDA_CMD="$exe -shape box 1 1 -size 2387.3241463784303 -lambda 500 -m 1.313 0.0 \
        -init_field zero -grid ${N} -eps 4.024 -iter ${solver} -pol fcd -int fcd -scat dr -ntheta 10"

      ADDA_STDOUT="$ADDA_LOG_DIR/adda_gpu_stdout_N=${N}_exe=${exe_name}_solver=${solver}_rep=${REP}.log"
      /usr/bin/time -v $ADDA_CMD 2>&1 | tee "$ADDA_STDOUT"

      # Handle ADDA GPU internal run directory: run*/log
      shopt -s nullglob
      run_dirs=( "$ADDA_DIR"/run*/ )
      shopt -u nullglob

      ADDA_RUN_LOG=""

      if ((${#run_dirs[@]} == 0)); then
        echo "WARNING: no run* directory found in $ADDA_DIR after ADDA GPU (N=${N}, exe=${exe_name}, solver=${solver}, rep=${REP})"
      else
        idx=$((${#run_dirs[@]} - 1))
        latest_run="${run_dirs[$idx]%/}"  # remove trailing slash
        log_file="${latest_run}/log"

        if [ -f "$log_file" ]; then
          ADDA_RUN_LOG="$ADDA_LOG_DIR/adda_gpu_log_N=${N}_exe=${exe_name}_solver=${solver}_rep=${REP}.log"
          echo "Moving ADDA GPU log file: $log_file -> $ADDA_RUN_LOG"
          mv "$log_file" "$ADDA_RUN_LOG"

          echo "Removing ADDA GPU run directory: $latest_run"
          rm -rf "$latest_run"
        else
          echo "WARNING: expected GPU log file not found: $log_file (N=${N}, exe=${exe_name}, solver=${solver}, rep=${REP})"
        fi
      fi

      #######################################################################
      # Extract metrics from ADDA GPU stdout and run log
      #######################################################################

      # From stdout:
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

      # From ADDA GPU log file:
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

      echo "${N},${exe_name},${solver},${REP},${first_residual},${last_residual},${cext_val},${qext_val},${elapsed_time},${total_iterations},${total_matvec},${total_wall_time},${solver_time},${fft_time}" >> "$ADDA_CSV"
      ;;

    #########################################################################
    # IFDDA GPU job: IFDDA N exe_name OMP rep
    #########################################################################
    IFDDA)
      N="$2"          # should always be 150 with current setup
      exe_name="$3"
      OMP="$4"
      REP="$5"

      echo
      echo "=============================="
      echo " IFDDA GPU job: N=${N}, exe=${exe_name}, OMP=${OMP}, rep=${REP}"
      echo "=============================="

      cd "$IFDDA_DIR"

      exe="./${exe_name}"
      if [ ! -x "$exe" ]; then
        echo "WARNING: executable $exe not found or not executable, skipping."
        continue
      fi

      export OMP_NUM_THREADS="$OMP"

      IFDDA_CMD="$exe -object cube 2387.3241463784303 -lambda 500 \
        -epsmulti 1.723969 0.0 \
        -ninitest 0 -nnnr ${N} -tolinit 9.46237161365793d-5 \
        -methodeit BICGSTAB -polarizability FG"

      IFDDA_STDOUT="$IFDDA_LOG_DIR/ifdda_gpu_stdout_N=${N}_exe=${exe_name}_omp=${OMP}_rep=${REP}.log"
      /usr/bin/time -v $IFDDA_CMD 2>&1 | tee "$IFDDA_STDOUT"

      # From stdout:
      res_lines=$(grep 'RESIDU' "$IFDDA_STDOUT" 2>/dev/null | grep -v 'INIT' || true)
      first_res_line=$(echo "$res_lines" | head -n1 || true)
      last_res_line=$(echo "$res_lines"  | tail -n1 || true)

      # GPU residual lines have the form:
      #   RESIDU 0.56081363048868715, Iteration: 1
      # -> 2nd field (strip trailing comma)
      first_residual=$(
        echo "$first_res_line" \
        | awk '{gsub(",","",$2); print $2}' || true
      )
      last_residual=$(
        echo "$last_res_line" \
        | awk '{gsub(",","",$2); print $2}' || true
      )

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

      echo "${N},${exe_name},${OMP},${REP},${first_residual},${last_residual},${cext_val},${elapsed_time},${num_iterations},${num_matvec},${total_time},${solver_time}" >> "$IFDDA_CSV"
      ;;

    *)
      echo "Unknown GPU job kind: $kind" >&2
      exit 1
      ;;
  esac

done

echo
echo "===== DDA GPU batch run finished: $(date) ====="
