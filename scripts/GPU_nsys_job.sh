#!/bin/bash
#BSUB -J nsys_cupy
#BSUB -q c02613
#BSUB -W 15
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=12GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "select[gpu80gb]"
#BSUB -o nsys_cupy_%J.out
#BSUB -e nsys_cupy_%J.err

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

DATA_DIR="/dtu/projects/02613_2025/data/modified_swiss_dwellings/"
N_FLOORPLANS=2 # reduce number of floorplans for profiling to keep runtime fast

nsys profile --stats=true -o cupy_profile --force-overwrite true \
    python src/simulate_cupy.py "$N_FLOORPLANS" "$DATA_DIR"