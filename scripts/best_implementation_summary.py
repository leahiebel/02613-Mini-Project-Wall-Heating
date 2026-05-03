import pandas as pd
import matplotlib.pyplot as plt

# Load the results
df = pd.read_csv("outputs/simulate_GPU_best_implementation_full_dataset.csv")
df.columns = df.columns.str.strip() 

# 1. Distribution of mean temperatures (Histogram)
df['mean_temp'].hist(bins=50, edgecolor='black')
plt.title("Distribution of Mean Temperatures")
plt.xlabel("Mean Temperature (°C)")
plt.ylabel("Count")
plt.show()
plt.savefig("outputs/mean_temp_distribution.png")

# 2. Average mean temperature
print("Avg Mean Temp:", df['mean_temp'].mean())

# 3. Average standard deviation
print("Avg Std Temp:", df['std_temp'].mean())

# 4. How many > 50% area above 18C?
buildings_above_18 = (df['pct_above_18'] >= 50).sum()
print(f"Buildings with >= 50% area > 18°C: {buildings_above_18}")

# 5. How many > 50% area below 15C?
buildings_below_15 = (df['pct_below_15'] >= 50).sum()
print(f"Buildings with >= 50% area < 15°C: {buildings_below_15}")