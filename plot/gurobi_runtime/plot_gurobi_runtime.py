import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme()

plt.rc('axes', labelsize=6, labelpad=2)
plt.rc('xtick', labelsize=5)
plt.rc('ytick', labelsize=5)
plt.rc('legend', fontsize=4.5)

width_mm = 90
height_mm = 50
in_per_mm = 0.03937008
plt.figure(figsize=(width_mm * in_per_mm, height_mm * in_per_mm))

dim_high = [201, 201, 201, 96]
dim_step = 5
sparsities = [5, 15, 25, 45]

for k, sparsity in enumerate(sparsities):
    dimensions = np.arange(sparsity, dim_high[k], dim_step)

    gurobi_runtime = np.load(os.path.join("runtime_data", f"gurobi_runtime_sparsity_{sparsity}.npy"))

    plt.plot(dimensions, np.mean(gurobi_runtime, axis=1), 'o-', markersize=2, label=f"Sparsity {sparsity}")

plt.legend(facecolor="white")
plt.title("Average Gurobi Runtime vs Dimension", fontsize=8)
plt.xlabel("Dimension")
plt.ylabel("Runtime (s)")
plt.yscale("log")
plt.tight_layout()

if not os.path.exists("figures"):
    os.mkdir("figures")
fname = "gurobi_runtime"
for ext in ['.svg', '.png']:
    plt.savefig(f"figures/{fname+ext}", dpi=300)
