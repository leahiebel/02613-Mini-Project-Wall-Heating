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

export PYTHONUNBUFFERED=1

PROJECT_ROOT="$PWD"
DATA_DIR="/dtu/projects/02613_2025/data/modified_swiss_dwellings/"
PROFILE_IDS_FILE="$PROJECT_ROOT/description/profile_subset_ids.txt"

SOLVER_MODULE="${1:-src/simulate_numba_cpu.py}"
SOLVER_FUNCTION="${2:-jacobi}"

echo "[INFO] Running Jacobi timing only"

python -u "$PROJECT_ROOT/scripts/profile_solver_numba.py" \
  --data-dir "$DATA_DIR" \
  --ids-file "$PROFILE_IDS_FILE" \
  --solver-module "$SOLVER_MODULE" \
  --solver-function "$SOLVER_FUNCTION" \
  --max-iter 20000 \
  --atol 1e-4