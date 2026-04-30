#!/bin/bash
#BSUB -J jacobi_profile
#BSUB -q hpc
#BSUB -W 15
#BSUB -R "rusage[mem=12GB]"
#BSUB -o job_outputs/%J.out
#BSUB -e job_outputs/%J.err
#BSUB -R "select[model == XeonGold6126]"
#BSUB -R "span[hosts=1]"
#BSUB -n 1

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026


PROJECT_ROOT="$PWD"
DATA_DIR="/dtu/projects/02613_2025/data/modified_swiss_dwellings/"
PROFILE_OUTPUT_DIR="$PROJECT_ROOT/outputs/profiling"
PROFILE_IDS_FILE="$PROJECT_ROOT/description/profile_subset_ids.txt"
MAX_ITER=20000
ATOL=1e-4
SOLVER_MODULE="${1:-src/simulate_numba_cpu.py}" #Just example, can be overwritten with args
SOLVER_FUNCTION="${2:-jacobi}" 
SOLVER_TARGET="$(basename "$SOLVER_MODULE" .py)"
PROFILE_TAG="$(basename "$SOLVER_MODULE" .py)_${SOLVER_FUNCTION}_profile_subset"
mkdir -p "$PROFILE_OUTPUT_DIR"


python -m kernprof -l -v \
  -s "$PROJECT_ROOT/scripts/kernprof_setup.py" \
  -p "$SOLVER_TARGET" \
  -o "$PROFILE_OUTPUT_DIR/${PROFILE_TAG}.lprof" \
  "$PROJECT_ROOT/scripts/profile_solver_numba.py" \
  --data-dir "$DATA_DIR" \
  --ids-file "$PROFILE_IDS_FILE" \
  --solver-module "$SOLVER_MODULE" \
  --solver-function "$SOLVER_FUNCTION" \
  --max-iter "$MAX_ITER" \
  --atol "$ATOL" | tee "$PROFILE_OUTPUT_DIR/${PROFILE_TAG}.txt"

