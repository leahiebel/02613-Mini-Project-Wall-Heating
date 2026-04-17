#!/bin/bash
#BSUB -J task3_viz
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
OUT_DIR="outputs/simulation_viz"

#Parameters to change
N_FLOORPLANS="${N_FLOORPLANS:-2}"
MAX_ITER="${MAX_ITER:-20000}"
ATOL="${ATOL:-1e-4}"


python scripts/visualize_simulation_results.py \
  --num "$N_FLOORPLANS" \
  --max-iter "$MAX_ITER" \
  --atol "$ATOL" \
  --data-dir "$DATA_DIR" \
  --out-dir "$OUT_DIR"