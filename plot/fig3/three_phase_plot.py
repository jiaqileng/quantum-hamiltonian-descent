import numpy as np
from numpy import sin, cos, exp, sqrt, pi
from scipy.sparse import diags, spdiags, identity, kron
from scipy.sparse.linalg import eigsh

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.ticker import LinearLocator
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import cm
import seaborn as sns

import os
from os import listdir
from os.path import isfile, isdir, join

import time
import h5py
import sys

sys.path.append(os.path.abspath("../../"))
from config import *


# Setup
STEPS = 256

SCATTER_SIZE = 0.5
TITLE_FONT = 12
TEXT_FONT = 10

TIMES = [0.1, 0.5, 1, 2, 5, 10]
RATIOS = [0.25, 0.2, 1.2]

# Levy things
# T1 = 0.5
# T1_idx = 5
# T2 = 3
# T2_idx = 14

plt.rcParams["text.usetex"] = True
sns.set_theme()

fname_to_label = {
    "dropwave": "Drop-Wave",
    "levy": "Levy",
    "levy13": "Levy 13",
    "holder": "Holder Table",
    "ackley2": "Ackley 2",
    "bohachevsky2": "Bohachevsky 2",
    "rosenbrock": "Rosenbrock",
    "camel3": "Camel 3",
    "csendes": "Csendes",
    "defl_corr_spring": "Defl. Corr. Spring",
    "michalewicz": "Michalewicz",
    "easom": "Easom",
    "xinsheyang3": "Xin-She Yang 3",
    "alpine1": "Alpine 1",
    "griewank": "Griewank",
    "ackley": "Ackley",
    "rastrigin": "Rastrigin",
    "alpine2": "Alpine 2",
    "hosaki": "Hosaki",
    "shubert": "Shubert",
    "styblinski_tang": "Styblinski-Tang",
    "sumofsquares": "Sum of Squares"
}


# Set instance names
# instances = ["ackley"]
instances = list(fname_to_label.keys())

for instance in instances:
    qhd_wfn_dir = f"{DATA_DIR_2D}/{instance}/{instance}_QHD256_WFN"

    if not os.path.isfile(f"./three_phase_data/{instance}_three_phase_data.npz"):
        continue

    with np.load(f"./three_phase_data/{instance}_three_phase_data.npz") as data:
        snapshot_times = data["snapshot_times"]
        qhd_prob_in_nbhd = data["qhd_prob_in_nbhd"]
        prob_spec = data["prob_spec"]
        qaa_snapshot_times = data["qaa_snapshot_times"]
        qaa_prob_in_nbhd = data["qaa_prob_in_nbhd"]
        E_ratio_snapshot_times = data["E_ratio_snapshot_times"]
        E_ratio_full = data["E_ratio_full"]


    X = np.linspace(0, 1-(1/STEPS), STEPS)
    Y = np.linspace(0, 1-(1/STEPS), STEPS)
    X, Y = np.meshgrid(X,Y)

    fig = plt.figure(figsize=(12, 6.5))
    gs = fig.add_gridspec(3, 6)


    # QUANTUM PROBABILITY SUBPLOT
    ax = fig.add_subplot(gs[0, :])
    ax.set_title(f"Quantum Probability Density ({fname_to_label[instance]})", fontsize=TITLE_FONT)
    ax.set_facecolor("white")
    ax.set_xticks([])
    ax.set_yticks([])

    for col in range(len(TIMES)):
        t = TIMES[col]
        wave_fn = np.load(os.path.join(qhd_wfn_dir, f"psi_{int(10*t)}e-01.npy"))
        density = (wave_fn * wave_fn.conj()).real

        ax = fig.add_subplot(gs[0, col], projection="3d")
        ax.set_facecolor("white")

        surf = ax.plot_surface(X, Y, density,
                               alpha=0.8,
                               cmap="cool",
                               linewidth=0,
                               antialiased=True)

        ax.text2D(0.35, 0.85, f"t={t}", transform=ax.transAxes)

        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xticklabels([])

        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_yticklabels([])

        ax.zaxis.set_major_formatter("{x:.2e}")
        ax.tick_params('z', labelsize=5, pad=0)


    # PROBABILITY SPECTRUM SUBPLOT
    ax = fig.add_subplot(gs[1, 0:3])
    ax.set_title("Probability Spectrum", fontsize=TITLE_FONT)

    im = ax.imshow(prob_spec,
               alpha=0.8,
               extent=[0, 125, 0, 10],
               origin="lower",
               interpolation="nearest",
               cmap="gist_heat_r")

    width_per_cell = 125 / len(snapshot_times)

    # ax.axvline(x=(T1_idx+0.5)*width_per_cell, linewidth=1, color="r",linestyle="dotted")
    # ax.axvline(x=(T2_idx+0.5)*width_per_cell, linewidth=1, color="r",linestyle="dotted")

    ax.grid(False)

    xtick_idx = np.array([0, 5, 10, 14, 18, 23, 28])
    ax.set_xticks((xtick_idx + 0.5) * width_per_cell)
    ax.set_xticklabels(snapshot_times[xtick_idx], fontsize=TEXT_FONT)
    ax.xaxis.set_ticks_position("bottom")

    yticks_loc = np.arange(1, 11) - 0.5
    ax.set_yticks(yticks_loc, labels=np.arange(0, 10))
    ax.yaxis.set_ticks_position("left")

    ax.set_xlabel("Time (t)", fontsize=TEXT_FONT)
    ax.set_ylabel("Energy Level", fontsize=TEXT_FONT)

    cbaxes = inset_axes(ax,
                        height="100%",
                        width="100%",
                        loc="lower right",
                        bbox_to_anchor=(0.9, 0.3, 0.03, 0.5),
                        bbox_transform=ax.transAxes)

    plt.colorbar(im,
                 cax=cbaxes,
                 ticks=[0, 0.5, 0.9, 1],
                 orientation="vertical")

    ax.set_aspect(125/10 * RATIOS[0])


    # SUCCESS PROBABILITY SUBPLOT
    ax = fig.add_subplot(gs[2, 0:3])
    ax.set_title("Success Probability", fontsize=TITLE_FONT)

    ax.plot(snapshot_times, 100 * prob_spec[0,:],
    	    linewidth=2,
            color="saddlebrown",
    	    label="QHD: overlap with ground state",
    	    marker='o',
            markersize=3)

    ax.plot(snapshot_times, 100 * qhd_prob_in_nbhd,
    	    linewidth=2,
            color="darkorange",
    	    label="QHD success rate",
    	    marker='o',
            markersize=3)

    ax.plot(qaa_snapshot_times, 100 * qaa_prob_in_nbhd,
    	    linewidth=2,
            color="dodgerblue",
    	    label="QAA success rate",)

    # ax.axvline(x=T1, linewidth=1, color="r", linestyle="dotted")
    # ax.axvline(x=T2, linewidth=1, color="r", linestyle="dotted")

    ax.set_xlim([0, 10])
    ax.set_xticks(np.arange(0, 12, 2))
    ax.set_xticklabels(np.arange(0, 12, 2), fontsize=TEXT_FONT)
    ax.xaxis.set_ticks_position("bottom")

    ax.set_xlabel("Time (t)", fontsize=TEXT_FONT)

    ax.set_yticks(np.arange(0, 101, step=25))
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=TEXT_FONT)
    ax.yaxis.grid(color="gray", linestyle="dashed")
    ax.set_axisbelow(True)

    ax.legend(loc="lower right",
              borderpad=1,
              labelspacing=0.8,
              fontsize=7,
              bbox_to_anchor=(1, 0.1))

    ax.set_aspect(10/110 * RATIOS[1])


    # ENERGY RATIO SUBPLOT
    ax = fig.add_subplot(gs[1:3, 3:6])
    ax.set_title(r"Energy Ratio $E_1/E_0$", fontsize=TITLE_FONT)

    ax.plot(E_ratio_snapshot_times, E_ratio_full,
        linewidth=2,
        color="slateblue",
        marker='o')

    ax.set_xlim([0, 20])
    ax.set_xticks(np.arange(0, 21, 1))
    ax.set_xticklabels(np.arange(0, 21, 1), fontsize=TEXT_FONT)

    ax.set_yticks([1, 1.5, 2, 2.5])
    ax.set_yticklabels([1, 1.5, 2, 2.5], fontsize=TEXT_FONT)
    ax.yaxis.tick_right()
    ax.yaxis.set_ticks_position("none")
    ax.yaxis.grid(color="gray", linestyle="dashed")

    ax.set_xlabel("Time (t)", fontsize=TEXT_FONT)

    ax.set_aspect(10/1.3 * RATIOS[2])


    #plt.show()
    plt.savefig(f"figures/test_{instance}_three_phase.png", dpi=150)
    # plt.savefig(f"figures/{instance}_three_phase.svg")
