import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from collections import Counter


mpl.style.use("seaborn")


def qp(x, Q, b):
    return (0.5 * x.T @ Q @ x) + (b.T @ x)


def count_close(samples, obj_fn, gnd_truth, obj_params):
    num_samples = samples.shape[0]
    count = 0

    for sample_idx in range(num_samples):
        x = np.transpose(samples[sample_idx, :])

        obj = obj_fn(x, *obj_params)

        if np.isclose(obj, gnd_truth, atol=1e-2):
            count += 1

    return count

DATA_DIR_QP = "/Users/ethan/LocalResearchData/HamiltonianDescent/QHD_DATA/QP"


DIMS = [50, 60, 75]
DURTNS = ["1e2", "2e2", "3e2", "4e2", "5e2", "6e2", "7e2", "8e2", "9e2", "1e3"]
NUM_EXPS = 50

data = {}

for dim in DIMS:
    print(dim)
    data[dim] = {}

    data[dim]["embedded-timed-processed"] = {}
    for timing in DURTNS:
        data[dim]["embedded-timed-processed"][timing] = []

    data[dim]["quantum-processed"] = []

    for i in range(NUM_EXPS):
        print(f"instance {i}")
        inst_data_dir = DATA_DIR_QP + f"/QP-{dim}d-5s/instance_{i}/"

        # Load QP
        source = inst_data_dir + f"instance_{i}.npy"

        with open(source, "rb") as f:
            Q = np.load(f)
            b = np.load(f)
            Q_c = np.load(f)
            b_c = np.load(f)

        qp_params = [Q, b]


        # Get quadratic program optimum from Gurobi
        fopt = inst_data_dir + f"instance_{i}_gurobi.npy"

        x_opt = np.load(fopt)
        qp_optimal_obj = qp(x_opt, Q, b)


        # Compare processed timed classical embedded model samples to QP optimum
        for dur in DURTNS:
            fsamples = inst_data_dir + f"post_timed_{dur}_sweeps_advantage6_classicaldwave_embedded_qhd_rez8_sample_{i}.npy"

            samples = np.load(fsamples)
            count = count_close(samples, qp, qp_optimal_obj, qp_params)
            data[dim]["embedded-timed-processed"][dur].append(count / samples.shape[0])

        # Compare processed DWave samples to QP optimum
        fsamples = inst_data_dir + f"post_advantage6_qhd_rez8_sample_{i}.npy"

        samples = np.load(fsamples)
        count = count_close(samples, qp, qp_optimal_obj, qp_params)
        data[dim]["quantum-processed"].append(count / samples.shape[0])
    print()


medians = {}
errs = {}

for dim in DIMS:
    df = pd.DataFrame.from_dict(data[dim]["embedded-timed-processed"])

    medians[dim] = []
    errs[dim] = np.ndarray((2, len(DURTNS)))

    for col_idx in range(len(df.columns)):
        col = df.columns[col_idx]
        medians[dim].append(np.median(df[col]))
        errs[dim][:, col_idx] = np.abs(np.quantile(df[col], [0.25, 0.75]).T - medians[dim][-1])


cmap = sns.color_palette("colorblind")

f, ax = plt.subplots(figsize=(90/25.4, 50/25.4), dpi=300)


for dim_idx in range(len(DIMS)):
    dim = DIMS[dim_idx]
    color = cmap[dim_idx]
    paired_color = tuple([(channel + 0.5*(1-channel)) for channel in color])
    plt.axhline(np.median(data[dim]["quantum-processed"]), color=paired_color, label=f"DW d{dim}", linestyle="--", linewidth=1.5)

    plt.plot(np.arange(len(medians[dim])),
         medians[dim],
         label=f"Rotor d{dim}",
         color=cmap[dim_idx],
         linewidth=1.5,
         marker="o", markersize=4)


# plt.title(f"Median Success Probability vs Dimension")
plt.xlabel("Classical Time Limit (sweeps)", size=6)
plt.ylabel("Median Success Probability", size=6)
plt.ylim([0, 0.15])


plt.xticks(range(len(DURTNS)), DURTNS, size=6)
plt.yticks(size=5)

plt.legend(frameon=True, facecolor="white", borderpad=0.35, prop={'size': 5})
plt.savefig("./figures/DWClassicalMedians.png", bbox_inches='tight')
plt.savefig("./figures/DWClassicalMedians.eps", bbox_inches='tight')
plt.show()
