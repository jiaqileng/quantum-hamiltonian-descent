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

# snapshot_times = [0.1, 0.5, 2, 3, 5, 10]
snapshot_times = []
snapshot_times.extend(list(np.arange(0, 1, 0.2)))
snapshot_times.extend(list(np.arange(1, 10, 2)))
snapshot_times.append(10)

n_plots = len(snapshot_times)
numruns = 1000

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

dirs = list(fname_to_label.keys())

# for instance in dirs:
for instance in ["levy"]:
    DATA_DIR = join(DATA_DIR_2D, f"{instance}")
    QHD_WFN_DIR = join(DATA_DIR, f"{instance}_QHD256_WFN")

    # QHD Setup
    qhd_steps = 256
    rez = 1 / qhd_steps
    cells = np.linspace(0,1-rez, qhd_steps)
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


    # Plot
    methods = ['QHD256', 'QAA128', 'NAGD', 'SGD']
    SCATTER_SIZE = 0.5
    TITLE_FONT = 14
    TEXT_FONT = 12

    sns.set_theme()
    fig = plt.figure()
    gs = fig.add_gridspec(4, 6, hspace=0.45)

    fig.suptitle(f'{fname_to_label[instance]} Function')

    for row in range(4):
        ax = fig.add_subplot(gs[row, :])
        ax.set_title(methods[row], fontsize=TEXT_FONT)
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

            ax = fig.add_subplot(gs[row, col])
            ax.scatter(x_data, y_data, s=SCATTER_SIZE, c='darkslateblue')
            ax.set_xlim([0,1])
            ax.set_ylim([0,1])
            ax.set_xticks([0,0.25,0.5,0.75,1])
            ax.set_yticks([0,0.25,0.5,0.75,1])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            if row == 3:
                ax.set_xlabel(f"t={snapshot_time}", fontsize=TEXT_FONT)

    plt.savefig(f'./figures/scatterplots/test_{instance}_scatter.png', bbox_inches='tight', dpi=300)
    # plt.savefig(f'./figures/scatterplots/{instance}_scatter.eps', bbox_inches='tight', dpi=300)
