import pandas as pd
import matplotlib.pyplot as plt

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

df = pd.read_csv("outputs/parallel_static_times.csv")

summary = df.groupby("workers")["elapsed_seconds"].mean().reset_index()

T1 = summary.loc[summary["workers"] == 1, "elapsed_seconds"].values[0]
summary["speedup"] = T1 / summary["elapsed_seconds"]

print(summary)

plt.figure()
plt.plot(summary["workers"], summary["speedup"], marker="o")
plt.xlabel("Number of workers")
plt.ylabel("Speedup")
plt.title("Speedup vs Workers (Static Scheduling)")
plt.grid()
plt.savefig("outputs/5.1_parallel_speedup_static.png")