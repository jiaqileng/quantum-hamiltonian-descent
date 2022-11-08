import numpy as np
import os, io
from os.path import join
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
plt.subplots_adjust(left=0.0, right=1.0, hspace=0.3)

cmap = sns.color_palette("colorblind")

DIMS = [5, 50, 60, 75]
plot_assignment = {5: (0,0), 50: (0, 1), 60: (1, 0), 75: (1, 1)}

for dim_idx in range(len(DIMS)):
    dim = DIMS[dim_idx]
    if dim == 5:
        benchmark_name = f"QP-{dim}d"
    else:
        benchmark_name = f"QP-{dim}d-5s"
    
    dataframe_filename = f"{benchmark_name}.xlsx"
    tab_prob = pd.read_excel(join("qp_data", dataframe_filename), sheet_name="time-to-solution (tts)", index_col=0)
    tab_prob.replace(np.inf, np.nan, inplace=True)
    num_methods = len(tab_prob.columns)

    sns.boxplot(ax=ax[plot_assignment[dim]], orient="h", data=tab_prob, palette="colorblind", showfliers=False, fliersize=FLIER_SIZE, linewidth=LINEWIDTH)
    
    means[dim] = []
    medians[dim] = []
    errs[dim] = np.ndarray((2, num_methods))
    
    for col_idx in range(num_methods):
        col = tab_prob.columns[col_idx]    
        
        col_mean = np.nanmean(tab_prob[col])
        means[dim].append(col_mean)
        
        box_top = len(tab_prob.columns)-col_idx
        box_center = (len(tab_prob.columns)-col_idx - 0.5)/len(tab_prob.columns)
        box_width = 8/60
        # plot tts mean as dotted vertical line per method
        #ax[plot_assignment[dim]].axvline(x=col_mean, ymin=(box_center-box_width/2), ymax=(box_center+box_width/2), linestyle=":", color="k", linewidth=1)
        
        medians[dim].append(np.nanmedian(tab_prob[col]))
        errs[dim][:, col_idx] = np.abs(np.quantile(tab_prob[col], [0.25, 0.75]).T - medians[dim][-1])
        
    ax[plot_assignment[dim]].axvline(medians[dim][0], linewidth=LINEWIDTH, color=cmap[0])

i = 0
for dim in [5, 50, 60, 75]:
    color = cmap[i]
    paired_color = tuple([(channel + 0.5*(1-channel)) for channel in color])
    i += 1


for key in plot_assignment.keys():
    ax[plot_assignment[key]].set_title(f"{key}d", size=XLABEL_SIZE, pad=2)
    ax[plot_assignment[key]].xaxis.set_tick_params(pad=2)
    ax[plot_assignment[key]].set_xlabel("time-to-solution (second)", size=XLABEL_SIZE)
    ax[plot_assignment[key]].set_xscale("log")
    ax[plot_assignment[key]].xaxis.labelpad = 2
    ax[plot_assignment[key]].tick_params(axis='x', which='major', labelsize=XTICKLABEL_SIZE)
    ax[plot_assignment[key]].tick_params(axis='y', which='major', labelsize=YTICKLABEL_SIZE)


if SHOULD_SAVE:
    if not os.path.exists("figures"):
        os.mkdir("figures")

    fname = "QP_TTS"
    for ext in ['.svg', '.png']:
        plt.savefig(f"figures/{fname+ext}")

plt.show()
