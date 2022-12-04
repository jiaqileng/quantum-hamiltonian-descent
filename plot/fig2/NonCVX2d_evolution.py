import numpy as np
from numpy import sin, pi
from scipy.io import loadmat
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
import seaborn as sns
import h5py

import os
from os import listdir
from os.path import isfile, isdir, join
from itertools import product
import sys

sys.path.append(os.path.abspath('../../'))
from config import *

def get_sample(coord_list, dist, numruns):    
    buffer = random.choices(coord_list, weights=dist, k=numruns)
    samples = np.zeros((numruns, 2))
    for j in range(numruns):
        samples[j] = buffer[j]
    return samples

def get_frame(qaa_times, snapshot_time):
    idx = 0
    while qaa_times[idx] <= snapshot_time:
        idx += 1
    return idx - 1



instance = "levy"
DATA_DIR = join(DATA_DIR_2D, f"{instance}")
QHD_WFN_DIR = join(DATA_DIR, f"{instance}_QHD256_WFN")

snapshot_times = [0.1, 0.5, 2, 3, 5, 10]
n_plots = len(snapshot_times)
numruns = 1000



# QHD Setup
steps = 256
rez = 1 / steps
cells = np.linspace(0,1-rez, steps)
qhd_coord_list = []
for i,j in product(cells, cells):
    qhd_coord_list.append([i,j])
    
if isdir(QHD_WFN_DIR) == False:
    print(f"The data path for function {instance} is not found.")



# QAA Setup
adb_steps = 128
dx = 1/(adb_steps+1)
adb_cells = np.linspace(dx, 1-dx, adb_steps)
qaa_coord_list = []

for i,j in product(adb_cells, adb_cells):
    qaa_coord_list.append([i,j])
    
QAA_DATA = {}
filepath = join(DATA_DIR, f"{instance}_QAA{adb_steps}_T10.mat")
f = h5py.File(filepath)
for k, v in f.items():
    QAA_DATA[k] = np.array(v)

qaa_times = QAA_DATA["snapshot_times"][0]
qaa_wfn = QAA_DATA["wfn"].view(np.complex128)

qaa_frames = []
for t in snapshot_times:
    qaa_frames.append(get_frame(qaa_times, t))



# NAGD Setup
NAGD_DATA = {}
loadmat(join(DATA_DIR, f"{instance}_NAGD.mat"), mdict=NAGD_DATA)
nagd_paths = NAGD_DATA['nesterov_positions']



# SGD Setup
SGD_DATA = {}
loadmat(join(DATA_DIR, f"{instance}_SGD.mat"), mdict=SGD_DATA)
sgd_paths = SGD_DATA['ngd_positions']



# Levy surface data
w = lambda v: 1 + (1/4)*(v - 1)
levy_fn = lambda x1, x2: sin(pi * w(x1))**2 + (w(x1) - 1)**2 * (1 + 10*sin(pi*w(x1) + 1)**2) + (w(x2) - 1)**2 * (1 + sin(2*pi*w(x2))**2)
L = 10
X, Y = np.meshgrid(np.arange(-L, L, 2*L/256), np.arange(-L, L, 2*L/256))
V = np.zeros((256,256))
for i in range(256):
    for j in range(256):
        V[i,j] = levy_fn(X[i,j],Y[i,j])
V_normalized = V/(2*L)
XX, YY = np.meshgrid(np.arange(0, 1, 1/256), np.arange(0, 1, 1/256))



# Plot 
methods = ['QHD256', 'QAA128', 'NAGD', 'SGD']
SCATTER_SIZE = 0.5
TITLE_FONT = 14
TEXT_FONT = 12
CMAP = 'cool'

sns.set_theme()
fig = plt.figure(figsize=(16.5, 8))
gs = fig.add_gridspec(4, 8)

ax1 = fig.add_subplot(gs[0:2, 0:2], projection='3d')
ax1.set_title('Levy function', fontsize=TITLE_FONT)
ax1.plot_surface(XX, YY, V_normalized, rstride=1, cstride=1, alpha=0.75, linewidths=0, cmap=CMAP)
ax1.contour(XX, YY, V_normalized, [5e-2, 1e-1, 5e-1, 1, 2.5, 4], cmap=CMAP, offset=-1)
ax1.contour(XX, YY, V_normalized, [5e-2, 1e-1, 5e-1, 1, 2.5, 4], colors='k', linewidths=1)
ax1.set_xticks([0,0.25,0.5,0.75,1])
ax1.set_xticklabels(['0','','0.5','',''], fontsize=7)
ax1.set_yticks([0,0.25,0.5,0.75,1])
ax1.set_yticklabels(['','','0.5','','1'], fontsize=7)
ax1.set_zticklabels([])

ax2 = fig.add_subplot(gs[2:4, 0:2])
ax2.set_title('heatmap', fontsize=TITLE_FONT)
ax2.grid(visible=None)
ax2.imshow(V_normalized, cmap=CMAP, origin='lower')
ax2.plot([.55*256-1],[.55*256-1], marker='*', markersize=12, mfc='darkslateblue', mec='white')
ax2.annotate('global \nminimizer', (.55*256, .55*256),
            xytext=(0.75, 0.8), textcoords='axes fraction',
            arrowprops=dict(facecolor='darkslateblue', shrink=0.05),
            fontsize=TEXT_FONT,
            horizontalalignment='left', verticalalignment='top')
ax2.set_xticks([0,63,127,191,255])
ax2.set_yticks([0,63,127,191,255])
ax2.set_xticklabels(['0','0.25','0.5','0.75','1'])
ax2.set_yticklabels(['0','0.25','0.5','0.75','1'])


for row in range(4):
    ax = fig.add_subplot(gs[row, 3:-1])
    ax.set_title(methods[row], fontsize=TITLE_FONT)
    ax.set_facecolor('white')
    ax.set_xticks([])
    ax.set_yticks([])

for col in range(n_plots):
    snapshot_time = snapshot_times[col]
    for row in range(4):
        if row == 0: # QHD
            wfn_fname = f'psi_{int(10*(snapshot_time))}e-01.npy'
            psi = np.load(os.path.join(QHD_WFN_DIR, wfn_fname))
            qhd_prob = (psi * psi.conj()).real.flatten('F')
            samples = get_sample(qhd_coord_list, qhd_prob, numruns)
            x_data = samples[:,0]
            y_data = samples[:,1]
            
        elif row == 1: # QAA
            frame = qaa_frames[col]
            wfn_frame = qaa_wfn[:,frame]
            qaa_prob = (wfn_frame * np.conj(wfn_frame)).real
            samples = get_sample(qaa_coord_list, qaa_prob, numruns)
            x_data = samples[:,0]
            y_data = samples[:,1]
            
        elif row == 2: # NAGD
            frame = int(snapshot_time/1e-2) - 1
            samples = np.zeros((numruns, 2))
            for k in range(numruns):
                samples[k,:] = nagd_paths[k][frame]
            x_data = samples[:,0]
            y_data = samples[:,1]
            
        elif row == 3: # SGD
            frame = int(snapshot_time / 1e-2) - 1
            x_data = sgd_paths[:, frame, 0]
            y_data = sgd_paths[:, frame, 1]
            
        ax = fig.add_subplot(gs[row, 2+col])
        ax.scatter(x_data, y_data,s=SCATTER_SIZE, c='darkslateblue')
        ax.set_xlim([0,1])
        ax.set_ylim([0,1])
        ax.set_xticks([0,0.25,0.5,0.75,1])
        ax.set_yticks([0,0.25,0.5,0.75,1])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        if row == 3:
            ax.set_xlabel(f"t={snapshot_time}", fontsize=TEXT_FONT)

plt.savefig('./figures/Evolution2D.png', dpi=300)
plt.savefig('./figures/Evolution2D.eps')
plt.show()
