#!/bin/bash
#BSUB -J CuPy
#BSUB -q c02613
#BSUB -W 15
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=12GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "select[gpu80gb]"
#BSUB -o CuPy_%J.out
#BSUB -e CuPy_%J.err

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

DATA_DIR="/dtu/projects/02613_2025/data/modified_swiss_dwellings/"
CSV_OUTPUT_DIR="outputs"
N_FLOORPLANS=5

# allow overrides
CANDIDATE_MODULE="${1:-src/simulate_cupy.py}"
CANDIDATE_SOLVER="${2:-jacobi}"

echo "Running validation..."

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

echo "Validation passed. Running GPU simulation..."

MODULE_NAME=$(basename "$CANDIDATE_MODULE" .py)
python "$CANDIDATE_MODULE" "$N_FLOORPLANS" "$DATA_DIR" \
  > "$CSV_OUTPUT_DIR/gpu_results_${MODULE_NAME}_${N_FLOORPLANS}.csv"