#!/bin/bash
#BSUB -q c02613
#BSUB -J gpujob
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=1GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 00:30
#BSUB -R "select[gpu80gb]"
#BSUB -o job_outputs/%J.out
#BSUB -e job_outputs/%J.err


source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026
DATA_DIR="/dtu/projects/02613_2025/data/modified_swiss_dwellings/"
CSV_OUTPUT_DIR="outputs"
#max 4571
N_FLOORPLANS=20

python src/simulate.py "$N_FLOORPLANS" "$DATA_DIR" > "$CSV_OUTPUT_DIR/gpu_results_${N_FLOORPLANS}.csv"