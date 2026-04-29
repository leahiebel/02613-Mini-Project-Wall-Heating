#!/bin/bash
#BSUB -J final_benchmark
#BSUB -q gpuv100
#BSUB -W 60
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "select[gpu80gb]"
#BSUB -o job_outputs/final_benchmark_%J.out
#BSUB -e job_outputs/final_benchmark_%J.err

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

DATA_DIR="/dtu/projects/02613_2025/data/modified_swiss_dwellings/"
CSV_OUTPUT_DIR="outputs"
PROFILE_OUTPUT_DIR="outputs/profiles"

mkdir -p "$CSV_OUTPUT_DIR"
mkdir -p "$PROFILE_OUTPUT_DIR"

MODULE="src/simulate_GPU_best_implementation.py"
MODULE_NAME=$(basename "$MODULE" .py)

echo ""
echo " PHASE 1: FULL DATASET (4571 FLOORPLANS)"
echo ""
N_ALL=4571

# The timing will print to the .err/.out files and not interfere with the CSV output
python "$MODULE" "$N_ALL" "$DATA_DIR" > "$CSV_OUTPUT_DIR/${MODULE_NAME}_full_dataset.csv"

echo ""
echo " PHASE 2: NSYS PROFILING (20 FLOORPLANS)"
echo ""
N_PROFILE=20

# We run nsys on a tiny subset just to capture the performance metrics and trace
nsys profile \
    --stats=true \
    --force-overwrite true \
    -o "$PROFILE_OUTPUT_DIR/${MODULE_NAME}_profile" \
    python "$MODULE" "$N_PROFILE" "$DATA_DIR" > /dev/null