#!/bin/bash
#BSUB -J python
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
DATA_DIR="/dtu/projects/02613_2025/data/modified_swiss_dwellings/"
CSV_OUTPUT_DIR="outputs"
N_FLOORPLANS=5

# Can override this with args when submitting job, just example:
CANDIDATE_MODULE="${1:-src/simulate_numba_cpu.py}"
CANDIDATE_SOLVER="${2:-jacobi}"

python scripts/validate_against_reference.py \
  --candidate-module "$CANDIDATE_MODULE" \
  --candidate-solver "$CANDIDATE_SOLVER" \
  --max-iter 20000 \
  --atol 1e-4
VALIDATION_EXIT=$?
if [ $VALIDATION_EXIT -ne 0 ]; then
  echo ""
  echo "ERROR: Validation failed! Candidate implementation differs from reference."
  exit 1
fi

# If validation passed, run candidate implementation on N_FLOORPLANS
# N_FLOORPLANS is not used in validation tests 
# Change candidate_results to match name of solver, otherwise mess
python "$CANDIDATE_MODULE" "$N_FLOORPLANS" "$DATA_DIR" > "$CSV_OUTPUT_DIR/candidate_results_${N_FLOORPLANS}.csv"

