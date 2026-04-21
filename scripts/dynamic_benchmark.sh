#!/bin/bash
#BSUB -J wallheat_dynamic
#BSUB -q hpc
#BSUB -n 16
#BSUB -W 02:00
#BSUB -R "span[hosts=1] rusage[mem=8000]"
#BSUB -o job_outputs/dynamic_%J.out
#BSUB -e job_outputs/dynamic_%J.err

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

cd ~/02613/project/02613-Mini-Project-Wall-Heating

mkdir -p outputs
mkdir -p job_outputs

N=20
REPEATS=5
WORKERS="1 2 4 8 16"

echo "workers,run,elapsed_seconds" > outputs/dynamic_times.csv

for W in $WORKERS
do
  for R in $(seq 1 $REPEATS)
  do
    python src/simulate_parallel_dynamic.py $N $W > /dev/null 2> tmp.err
    T=$(grep "ELAPSED_SECONDS=" tmp.err | cut -d= -f2)
    echo "$W,$R,$T" >> outputs/dynamic_times.csv
  done
done

rm -f tmp.err
echo "Done."
