#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Project setup
###############################################################################

PROJECT_ROOT="/home/argentic@coria.fr/Bureau/Work/Data_DDA_benchmark_paper"

ADDA_DIR="$PROJECT_ROOT/adda/src/ocl"
IFDDA_DIR="$PROJECT_ROOT/if-dda/tests/test_command"

###############################################################################
# Logging setup (global script log) - GPU
###############################################################################

LOG_ROOT="$PROJECT_ROOT/logs_GPU"
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
  echo "N,exe,solver,first_residual,last_residual,Cext,Qext,elapsed_time,total_iterations,total_matvec,total_wall_time,solver_time,fft_time" > "$ADDA_CSV"
fi

if [ ! -f "$IFDDA_CSV" ]; then
  echo "N,exe,OMP,first_residual,last_residual,Cext,elapsed_time,num_iterations,num_matvec,total_time,solver_time" > "$IFDDA_CSV"
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
# GPU runs: ADDA for N=150 and 250, IFDDA only for N=150
###############################################################################

for N in 150 250; do
  echo
  echo "=============================="
  echo "        GPU runs, N = ${N}"
  echo "=============================="

  ###########################################################################
  # ADDA GPU: ./adda_ocl and ./adda_ocl_blas
  # - adda_ocl runs with solvers: bicgstab and bicg
  # - adda_ocl_blas runs with solver: bicg
  ###########################################################################
  echo
  echo "### ADDA GPU runs, N=${N}"
  cd "$ADDA_DIR"

  # OpenMP is usually irrelevant for GPU, but keep it explicit
  export OMP_NUM_THREADS=1

  for exe in ./adda_ocl ./adda_ocl_blas; do
    if [ ! -x "$exe" ]; then
      echo "WARNING: executable $exe not found or not executable, skipping."
      continue
    fi

    # Choose solvers depending on executable
    if [ "$exe" = "./adda_ocl" ]; then
      solvers=("bicgstab" "bicg")
    else
      solvers=("bicg")
    fi

    exe_name=$(basename "$exe")

    for solver in "${solvers[@]}"; do
      echo
      echo "---- ADDA GPU: exe=${exe_name}, solver=${solver}, N=${N} ----"

      ADDA_CMD="$exe -shape box 1 1 -size 2.3873241463784303 -lambda 0.5 -m 1.313 0.0 \
        -init_field zero -grid ${N} -eps 4.024 -iter ${solver} -pol fcd -int fcd -scat dr -ntheta 10"

      # Per-run stdout log
      ADDA_STDOUT="$ADDA_LOG_DIR/adda_gpu_stdout_N=${N}_exe=${exe_name}_solver=${solver}.log"
      /usr/bin/time -v $ADDA_CMD 2>&1 | tee "$ADDA_STDOUT"

      # After each ADDA GPU run:
      # - find the run directory (ADDA_DIR/run*/),
      # - move the file run*/log to ADDA_LOG_DIR,
      # - remove the run directory to avoid conflicts with next runs.

      shopt -s nullglob
      run_dirs=( "$ADDA_DIR"/run*/ )
      shopt -u nullglob

      ADDA_RUN_LOG=""

      if ((${#run_dirs[@]} == 0)); then
        echo "WARNING: no run* directory found in $ADDA_DIR after ADDA GPU (N=${N}, exe=${exe_name}, solver=${solver})"
      else
        # Take the last directory in the list (usually the most recent)
        idx=$((${#run_dirs[@]} - 1))
        latest_run="${run_dirs[$idx]}"
        latest_run="${latest_run%/}"  # remove trailing slash

        log_file="${latest_run}/log"  # file is always named 'run*/log'

        if [ -f "$log_file" ]; then
          ADDA_RUN_LOG="$ADDA_LOG_DIR/adda_gpu_log_N=${N}_exe=${exe_name}_solver=${solver}.log"
          echo "Moving ADDA GPU log file: $log_file -> $ADDA_RUN_LOG"
          mv "$log_file" "$ADDA_RUN_LOG"

          echo "Removing ADDA GPU run directory: $latest_run"
          rm -rf "$latest_run"
        else
          echo "WARNING: expected GPU log file not found: $log_file (N=${N}, exe=${exe_name}, solver=${solver})"
        fi
      fi

      #######################################################################
      # Extract metrics from ADDA GPU stdout and run log
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

      # From ADDA GPU log file:
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

      # Append to ADDA GPU CSV
      echo "${N},${exe_name},${solver},${first_residual},${last_residual},${cext_val},${qext_val},${elapsed_time},${total_iterations},${total_matvec},${total_wall_time},${solver_time},${fft_time}" >> "$ADDA_CSV"

    done  # solver
  done    # exe

  ###########################################################################
  # IFDDA GPU: ./ifdda_GPU_single and ./ifdda_GPU
  # - Both use BICGSTAB
  # - Run for OMP_NUM_THREADS = 1 and 10
  # - Only for N = 150
  ###########################################################################
  if [ "$N" -eq 150 ]; then
    echo
    echo "### IFDDA GPU runs, N=${N}"
    cd "$IFDDA_DIR"

    for exe in ./ifdda_GPU_single ./ifdda_GPU; do
      if [ ! -x "$exe" ]; then
        echo "WARNING: executable $exe not found or not executable, skipping."
        continue
      fi

      exe_name=$(basename "$exe")

      for OMP in 1 10; do
        echo
        echo "---- IFDDA GPU: exe=${exe_name}, N=${N}, OMP_NUM_THREADS=${OMP} ----"
        export OMP_NUM_THREADS="$OMP"

        IFDDA_CMD="$exe -object cube 2387.3241463784303 -lambda 500 \
          -epsmulti 1.7239689999999999 0.0 \
          -ninitest 0 -nnnr ${N} -tolinit 9.46237161365793d-5 \
          -methodeit BICGSTAB -polarizability FG"

        IFDDA_STDOUT="$IFDDA_LOG_DIR/ifdda_gpu_stdout_N=${N}_exe=${exe_name}_omp=${OMP}.log"
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

                res_lines=$(grep 'RESIDU' "$IFDDA_STDOUT" 2>/dev/null | grep -v 'INIT' || true)
        first_res_line=$(echo "$res_lines" | head -n1 || true)
        last_res_line=$(echo "$res_lines"  | tail -n1 || true)

        # GPU residual lines have the form:
        #   RESIDU 0.56081363048868715, Iteration: 1
        # -> value is the 2nd field, possibly ending with a comma
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

        # Fallback NA
        first_residual=$(safe_value "$first_residual")
        last_residual=$(safe_value "$last_residual")
        cext_val=$(safe_value "$cext_val")
        elapsed_time=$(safe_value "$elapsed_time")
        num_iterations=$(safe_value "$num_iterations")
        num_matvec=$(safe_value "$num_matvec")
        total_time=$(safe_value "$total_time")
        solver_time=$(safe_value "$solver_time")

        # Append to IFDDA GPU CSV
        echo "${N},${exe_name},${OMP},${first_residual},${last_residual},${cext_val},${elapsed_time},${num_iterations},${num_matvec},${total_time},${solver_time}" >> "$IFDDA_CSV"

      done  # OMP
    done    # exe
  fi  # end IFDDA GPU for N=150

done  # N

echo
echo "===== DDA GPU batch run finished: $(date) ====="
