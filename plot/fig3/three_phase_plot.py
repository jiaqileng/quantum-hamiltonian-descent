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


sys.path.append(os.path.abspath('../../'))
from config import *



###-----------------------------
# Load Plot Data
###-----------------------------
instance = 'levy'
steps = 256
QHD_WFN_DIR = f"{DATA_DIR_2D}/{instance}/{instance}_QHD256_WFN"
ADB_WFN_DIR = f"{DATA_DIR_2D}/{instance}/{instance}_QAA128_T10.mat"
fname = f"../fig3/{instance}_three_phase_data.npy"
with open(fname, 'rb') as f:
    snapshot_times = np.load(f)
    qhd_prob_in_nbhd = np.load(f)
    prob_spec = np.load(f)
    qaa_snapshot_times = np.load(f)
    qaa_prob_in_nbhd = np.load(f)
    E_ratio_snapshot_times = np.load(f)
    E_ratio_full = np.load(f)

X = np.linspace(0,1-1/steps,steps)
Y = np.linspace(0,1-1/steps,steps)
X, Y = np.meshgrid(X,Y)

###-----------------------------
# Plot Parameters
###-----------------------------
titles = ['Quantum Probability Density', 'Energy Spectrum', 'Success Probability', 'Energy Ratio']
SCATTER_SIZE = 0.5
TITLE_FONT = 12
TEXT_FONT = 10
CMAP = 'cool'

times = [0.1, 0.5, 1, 2, 5, 10]
ratio_0 = 1/4
ratio_1 = 1/5
ratio_2 = 1.2

T1 = 0.5
T1_idx = 5
T2 = 3
T2_idx = 14

sns.set_theme()
fig = plt.figure(figsize=(12, 6.5))
gs = fig.add_gridspec(3, 6)
###-----------------------------
# Quantum Probability
###-----------------------------
row = 0
ax = fig.add_subplot(gs[row, :])
ax.set_title('Quantum Probability Density (Levy)', fontsize=TITLE_FONT)
ax.set_facecolor('white')
ax.set_xticks([])
ax.set_yticks([])
for col in range(6):
    t = times[col]
    QHD_WFN_DIR = f"/Users/lengjiaqi/QHD_DATA/NonCVX-2d/{instance}/{instance}_QHD256_WFN"
    wave_fn = np.load(os.path.join(QHD_WFN_DIR, f"psi_{int(t*10)}e-01.npy"))
    density = (wave_fn * wave_fn.conj()).real
    ax = fig.add_subplot(gs[0, col], projection='3d')
    surf = ax.plot_surface(X, Y, density,
                           alpha = 0.8,
                           cmap=CMAP,
                           linewidth=0, 
                           antialiased=True)
    ax.text2D(0.35, 0.85, f"t={t}", transform=ax.transAxes)
    ax.set_xticks([0,0.25,0.5,0.75,1])
    ax.set_xticklabels([])
    ax.set_yticks([0,0.25,0.5,0.75,1])
    ax.set_yticklabels([])
    ax.set_zticklabels([])



###-----------------------------
# Probability Spectrum
###-----------------------------
row = 1
ax = fig.add_subplot(gs[row, 0:3])
ax.set_title("Probability Spectrum", fontsize=TITLE_FONT)
im = ax.imshow(prob_spec,
           alpha = 0.8,
           extent=[0,125,0,10],
           origin='lower',
           interpolation='nearest',
           cmap='gist_heat_r')
per_cell = 125/len(snapshot_times) # the width per cell on the heatmap
ax.axvline(x=(T1_idx+0.5)*per_cell, linewidth=1, color='r',linestyle='dotted')
ax.axvline(x=(T2_idx+0.5)*per_cell, linewidth=1, color='r',linestyle='dotted')
# disable gridlines
ax.grid(visible=None)
# x ticks
xtick_idx = np.array([0, 5,10,14,18,23,28])
ax.set_xticks((xtick_idx+0.5)*per_cell)
ax.set_xticklabels(snapshot_times[xtick_idx],fontsize=TEXT_FONT)
ax.xaxis.set_ticks_position('bottom')
# x label
ax.set_xlabel('Time (t)', fontsize=TEXT_FONT)
# y ticks
yticks_loc = np.arange(1,11) - 0.5
ax.set_yticks(yticks_loc, labels=np.arange(0,10))
ax.yaxis.set_ticks_position('left')
# y label
ax.set_ylabel('Energy Level', fontsize=TEXT_FONT)
# colorbar
cbaxes = inset_axes(ax,width="100%", height="100%",
                    loc='lower right',
                    bbox_to_anchor=(1-.1,.3,.03,.5),
                    bbox_transform=ax.transAxes) 
plt.colorbar(im, cax=cbaxes, 
             ticks=[0.,0.5,0.9,1.0], 
             orientation='vertical')
# plot aspect
ax.set_aspect(125/10*ratio_0)


        
###-----------------------------
# Success Probability
###-----------------------------
row = 2
ax = fig.add_subplot(gs[row, 0:3])
ax.set_title("Success Probability", fontsize=TITLE_FONT)
ax.plot(snapshot_times,100 * prob_spec[0,:], 
	    linewidth=2, color='saddlebrown', 
	    label="QHD: overlap with ground state",
	    marker = 'o', markersize=3)
ax.plot(snapshot_times, 100 * qhd_prob_in_nbhd,
	    linewidth=2, color='darkorange', 
	    label="QHD success rate",
	    marker = 'o', markersize=3)
ax.plot(qaa_snapshot_times,
	    100 * qaa_prob_in_nbhd,
	    linewidth=2, color='dodgerblue', 
	    label="QAA success rate",)
ax.axvline(x=T1, linewidth=1, color='r',linestyle='dotted')
ax.axvline(x=T2, linewidth=1, color='r',linestyle='dotted')
# x ticks
ax.set_xlim([0,10])
ax.set_xticks(np.arange(0,12,2))
ax.set_xticklabels(np.arange(0,12,2), fontsize=TEXT_FONT)
ax.xaxis.set_ticks_position('bottom')
# x label
ax.set_xlabel('Time (t)', fontsize=TEXT_FONT)
# y ticks
#ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_ylim([0,110])
ax.set_yticks(np.arange(0,101, step=25))
ax.set_yticklabels(['0%','25%','50%','75%','100%'], fontsize=TEXT_FONT)
ax.yaxis.grid(color='gray', linestyle='dashed')
ax.set_axisbelow(True)
# legend
ax.legend(loc="lower right",
           borderpad=1, 
           labelspacing=0.8, 
           fontsize=7,
           bbox_to_anchor=(1,0.1))
# plot aspect
ax.set_aspect(10/110*ratio_1)  
    


###-----------------------------
# Energy Ratio
###-----------------------------
ax = fig.add_subplot(gs[1:3, 3:6])
plt.rcParams['text.usetex'] = True
ax.set_title('Energy Ratio  ' + r"$E_1/E_0$", fontsize=TITLE_FONT)
ax.plot(E_ratio_snapshot_times, E_ratio_full, linewidth=2, color='slateblue', marker='o')
#ax.text(4, 2.5, "t = 0\n" + r"$\frac{E_1}{E_0} = 2.5$", fontsize=TEXT_FONT+2)
#ax.text(14, 2.1,"semi-classical limit\n" + r"$\lim_{t \to \infty}\frac{E_1}{E_0} \approx 1.38$", fontsize=TEXT_FONT+2)
# x ticks
ax.set_xlim([0,20])
ax.set_xticks(np.arange(0,20, step=1))
ax.set_xticklabels(np.arange(0,20, step=1), fontsize=TEXT_FONT)
# x label
ax.set_xlabel('Time (t)', fontsize=TEXT_FONT)
# y ticks
ax.set_ylim([1.4,2.7])
ax.set_yticks([1.5,2.,2.5])
ax.set_yticklabels([1.5,2.,2.5], fontsize=TEXT_FONT)
ax.yaxis.tick_right()
ax.yaxis.set_ticks_position('none')
ax.yaxis.grid(color='gray', linestyle='dashed')
# plot aspect
ax.set_aspect(10/1.3*ratio_2)


###-----------------------------
# Save Plot
###-----------------------------
#plt.show()
plt.savefig(f"figures/{instance}_three_phase.png", dpi=150)
plt.savefig(f"figures/{instance}_three_phase.svg")