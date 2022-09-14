import numpy as np
import os, io
from io import BytesIO
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import json

import seaborn as sns
mpl.style.use("seaborn")



SHOULD_SAVE = True



# Plot Params
SUPTITLE_SIZE = 7

XLABEL_SIZE = 6
XTICKLABEL_SIZE = 6

YLABEL_SIZE = 6
YTICKLABEL_SIZE = 5

# Box plot
BOX_WIDTH = 0.8
FLIER_SIZE = 2
LINEWIDTH = 0.75

# Stats plot
MARKER_SIZE = 5
LEGEND_SIZE = 6

means = {}
medians = {}
errs = {}

f, ax = plt.subplots(2, 2, figsize=(180/25.4, 95/25.4), dpi=300)
# f, ax = plt.subplots(2, 2, figsize=(180/25.4, 135/25.4), dpi=300)
# plt.subplots_adjust(left=0.0, right=1.0, hspace=0.1)
plt.subplots_adjust(left=0.0, right=1.0, hspace=0.3)

cmap = sns.color_palette("colorblind")

DIMS = [50, 60, 75]
plot_assignment = {50: (0, 0), 60: (0, 1), 75: (1, 0), "stats": (1, 1)}

for dim_idx in range(len(DIMS)):
    dim = DIMS[dim_idx]
    benchmark_name = f"QP-{dim}d-5s"
        

    tab_prob = pd.DataFrame(
        {
            "DW-QHD": np.load(os.path.join("qp_data", benchmark_name, f"post_advantage6_qhd_rez8_success_prob.npy")),
            "IPOPT": np.load(os.path.join("qp_data", benchmark_name, f"ipopt_success_prob.npy")),
            "DW-QAA": np.load(os.path.join("qp_data", benchmark_name, f"post_advantage6_qaa_scheduleB_rez8_success_prob.npy")),
            "MATLAB": np.load(os.path.join("qp_data", benchmark_name, f"matlab_sqp_success_prob.npy")),
            "SNOPT": np.load(os.path.join("qp_data", benchmark_name, f"snopt_success_prob.npy")),
            "AUGLAG": np.load(os.path.join("qp_data", benchmark_name, f"auglag_success_prob.npy")),
        }
    )

    sns.boxplot(ax=ax[plot_assignment[dim]], orient="h", data=tab_prob, palette="colorblind", showfliers=False, fliersize=FLIER_SIZE, linewidth=LINEWIDTH)
    
    means[dim] = []
    medians[dim] = []
    errs[dim] = np.ndarray((2, len(tab_prob.columns)))
    
    for col_idx in range(len(tab_prob.columns)):
        col = tab_prob.columns[col_idx]    
        
        col_mean = np.mean(tab_prob[col])
        means[dim].append(col_mean)
        
        box_top = len(tab_prob.columns)-col_idx
        box_center = (len(tab_prob.columns)-col_idx - 0.5)/len(tab_prob.columns)
        box_width = 8/60
        ax[plot_assignment[dim]].axvline(x=col_mean, ymin=(box_center-box_width/2), ymax=(box_center+box_width/2), linestyle=":", color="k", linewidth=1)
        
        medians[dim].append(np.median(tab_prob[col]))
        errs[dim][:, col_idx] = np.abs(np.quantile(tab_prob[col], [0.25, 0.75]).T - medians[dim][-1])
        
    ax[plot_assignment[dim]].axvline(medians[dim][0], linewidth=LINEWIDTH, color=cmap[0])
    
    ax[(0, 0)].set_title("d50", size=XLABEL_SIZE, pad=2)
    ax[(0, 1)].set_title("d60", size=XLABEL_SIZE, pad=2)
    ax[(1, 0)].set_title("d75", size=XLABEL_SIZE, pad=2)
    ax[(1, 1)].set_title("Summary Statistics", size=XLABEL_SIZE, pad=2)

    ax[(0, 0)].xaxis.set_tick_params(pad=2)
    ax[(0, 1)].xaxis.set_tick_params(pad=2)
    ax[(1, 0)].xaxis.set_tick_params(pad=2)
    ax[(1, 1)].xaxis.set_tick_params(pad=2)

    ax[(0, 0)].set_xlabel("Success Probability", size=XLABEL_SIZE)
    ax[(0, 1)].set_xlabel("Success Probability", size=XLABEL_SIZE)
    ax[(1, 0)].set_xlabel("Success Probability", size=XLABEL_SIZE)
    ax[(1, 1)].set_xlabel("Optimization Method", size=XLABEL_SIZE)
    
    ax[(0, 0)].xaxis.labelpad = 2
    ax[(0, 1)].xaxis.labelpad = 2
    ax[(1, 0)].xaxis.labelpad = 2
    ax[(1, 1)].xaxis.labelpad = 2

i = 0
for dim in [50, 60, 75]:
    color = cmap[i]
    paired_color = tuple([(channel + 0.5*(1-channel)) for channel in color])

    ax[plot_assignment["stats"]].errorbar(x=np.arange(len(medians[dim])), y=medians[dim], label=f"d{dim} Medians", marker="o", markersize=MARKER_SIZE, linestyle="-", linewidth=1, color=color)
    ax[plot_assignment["stats"]].errorbar(x=np.arange(len(means[dim])), y=means[dim], label=f"d{dim} Means", marker="o", markersize=MARKER_SIZE, linestyle="--", linewidth=1,  color=paired_color)
    i += 1

ax[plot_assignment["stats"]].set_xticks(range(6), tab_prob.columns)


for key in plot_assignment.keys():
    ax[plot_assignment[key]].tick_params(axis='x', which='major', labelsize=XTICKLABEL_SIZE)
    ax[plot_assignment[key]].tick_params(axis='y', which='major', labelsize=YTICKLABEL_SIZE)

ax[plot_assignment["stats"]].legend(bbox_to_anchor=(1, 1), frameon=True, facecolor="white", borderpad=0.35, prop={'size': LEGEND_SIZE})
ax[plot_assignment["stats"]].set_ylabel("Success Probability", size=YLABEL_SIZE)


if SHOULD_SAVE:
	if not os.path.exists("figures"):
		os.mkdir("figures")
	
	fname = "QPMegaplotThin"
	for ext in ['.svg', '.png']:
		plt.savefig(f"figures/{fname+ext}")

plt.show()