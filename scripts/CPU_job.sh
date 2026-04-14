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
#max 4571
N_FLOORPLANS=20

# Run simulation on CPU and save results to CSV
python src/simulate.py "$N_FLOORPLANS" "$DATA_DIR" > "$CSV_OUTPUT_DIR/cpu_results_${N_FLOORPLANS}.csv"