import numpy as np
from numpy import sin, cos, exp, sqrt, pi
from scipy.sparse import diags, spdiags, identity, kron
from scipy.sparse.linalg import eigsh
from scipy.io import loadmat

import os
from os import listdir
from os.path import isfile, isdir, join
import time
import h5py
import sys

import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('../../'))
from config import *


def is_point_within_RADIUS(point, target, RADIUS):
    dist2 = (target[0] - point[0])**2 + (target[1] - point[1])**2
    if dist2 <= RADIUS**2:
        return True
    return False

def grid_points_within_radius(center, RADIUS, grid_steps):
    support = np.linspace(1/(grid_steps+1), 1 - 1/(grid_steps+1), grid_steps)
    points = []

    # The rows count down since the origin is now in the bottom left
    # instead of the top left
    for c in range(grid_steps):
        for r in range(grid_steps):
            dist2 = (center[0] - support[c])**2 + (center[1] - support[-r-1])**2
            if dist2 < RADIUS**2:
                points.append((c, grid_steps-r-1))
    return [grid_steps*c + r for (c, r) in points]


# Setup
QHD_STEPS = 256
QAA_STEPS = 128

# experiment setup params
MAX_ENERGY_LEVEL = 10
RADIUS = 0.1
stepsize = 1/(QHD_STEPS+1)

global_min_locs = {
    "ackley": (0.5000,    0.5000),
    "ackley2": (0.5000, 0.5000),
    "alpine1": (0.5000,    0.5000),
    "alpine2": (0.7917,    0.7917),
    "bohachevsky2": (0.5000,    0.5000),
    "camel3": (0.5000,    0.5000),
    "csendes": (0.5000,    0.5000),
    "defl_corr_spring": (0.5000,    0.5000),
    "dropwave": (0.5000,    0.5000),
    "easom": (0.5000,    0.5000),
    "griewank": (0.5000,    0.5000),
    "holder": (0.8055,    0.9665),
    "hosaki": (0.8000,    0.4000),
    "levy": (0.5500,    0.5500),
    "levy13": (0.5500,    0.5500),
    "michalewicz": (0.7003,    0.4997),
    "rastrigin": (0.5000,    0.5000),
    "rosenbrock": (0.8333,    0.8333),
    "shubert": (0.3214,    0.7713),
    "styblinski_tang": (0.2097,    0.2097),
    "sumofsquares": (0.5000, 0.5000),
    "xinsheyang3": (0.5000, 0.5000)
}

# L = instance_bounds[1]-instance_bounds[0]
# dX = L / (QHD_STEPS + 1)

# meshpoints = np.linspace(instance_bounds[0]+dX, instance_bounds[1]-dX, QHD_STEPS)
# support = np.linspace(stepsize, 1-stepsize, QHD_STEPS)

# Construct the kinetic operator
e = np.ones(QHD_STEPS)
H0 = spdiags([e, -2*e, e], [-1, 0, 1], QHD_STEPS, QHD_STEPS)
Id = identity(QHD_STEPS);
H_T = -0.5 / (stepsize**2) * (kron(Id, H0) + kron(H0, Id))

# Load from potentials256.mat
potentials256 = loadmat(os.path.abspath("../../NonCVX-2d/experiments/QHD/potentials256.mat"))

# for instance_idx in range(len(potentials256["names"][0])):
instance_idcs = range(20, 22)
for instance_idx in instance_idcs:
    instance = potentials256["names"][0][instance_idx][0]


    data_dir = join(DATA_DIR_2D, f"{instance}")
    qhd_wfn_dir = join(data_dir, f"{instance}_QHD{QHD_STEPS}_WFN")
    qaa_wfn_dir = join(data_dir, f"{instance}_QAA{QAA_STEPS}_T10.mat")

    # Load the instance's potential operator
    square_potential = potentials256["potentials"][instance_idx]
    vector_potential = np.reshape(square_potential, (QHD_STEPS**2,), order='F')
    H_U = diags(vector_potential, 0)

    global_min_loc = global_min_locs[instance]


    # Compute the QHD data for the three-phase plot

    # Mask for items in the neighborhood
    nbhd_idcs = grid_points_within_radius(global_min_loc, RADIUS, QHD_STEPS)
    nbhd_locs = np.zeros((QHD_STEPS**2, 1))
    nbhd_locs[nbhd_idcs] = 1
    nbhd_locs = np.reshape(nbhd_locs, (QHD_STEPS, QHD_STEPS), order='F')

    if isdir(qhd_wfn_dir) == False:
        print(f"The WFN data path for function {instance} is not found.")
    else:
        # Give integer frame indices to compute based on.
        # Frame time = frame index / 10
        snapshot_idcs = np.concatenate((np.arange(0, 10, 1), np.arange(10, 105, 5)))
        num_frames = len(snapshot_idcs)
        snapshot_times = 0.1 * snapshot_idcs

        qhd_prob_in_nbhd = np.zeros(num_frames)
        prob_spec = np.zeros((MAX_ENERGY_LEVEL, num_frames))
        E_ratio = np.zeros(num_frames)

        print(f"Start working on {instance} function.")

        for j in range(num_frames):
            idx = snapshot_idcs[j]
            t = snapshot_times[j]
            print(f"QHD snapshot time = {t}.")
            if idx == 0:
                wfn_fname = f"psi_0.npy"
            else:
                wfn_fname = f"psi_{idx}e-01.npy"
            psi = np.load(os.path.join(qhd_wfn_dir, wfn_fname))

            # Compute QHD success probability at time t
            qhd_prob_in_nbhd[j] = (nbhd_locs * ((psi * psi.conj()).real)).sum()
            print(f"-- QHD success probability finished.")

            # Compute QHD probability spectrum at time t
            tdep1 = 2 / (1e-3 + t**3)
            tdep2 = 2 * t**3
            if t == 0:
                H = (tdep1 * H_T) + (tdep2 * H_U)
            else:
                H = (tdep1 / tdep2 * H_T) + H_U
            start = time.time()
            eigvals, eigvecs = eigsh(H, k=MAX_ENERGY_LEVEL, which='SA')
            end = time.time()
            spec = psi.flatten('F') @ eigvecs
            prob_spec[:,j] = (spec * spec.conj()).real
            print(f"-- QHD probability spectrum finished, matrix diagonalization time = {end-start}.")

            # Compute QHD energy ratio at time t
            E_ratio[j] = eigvals[1] / eigvals[0]
            print(f"-- QHD energy ratio finished.\n")


    # Compute QAA success probability data
    nbhd_idcs = grid_points_within_radius(global_min_loc, RADIUS, QAA_STEPS)
    nbhd_locs = np.zeros((QAA_STEPS**2, 1))
    nbhd_locs[nbhd_idcs] = 1
    nbhd_locs = np.reshape(nbhd_locs, (QAA_STEPS, QAA_STEPS), order='F')

    qaa_data = {}
    qaa_f = h5py.File(qaa_wfn_dir, 'r')
    for k, v in qaa_f.items():
        qaa_data[k] = np.array(qaa_f.get(k))

    qaa_wfn = qaa_data["wfn"].view(np.complex128)
    qaa_snapshot_times = qaa_data["snapshot_times"][0]
    qaa_num_frames = len(qaa_snapshot_times)
    qaa_prob_in_nbhd = np.empty(qaa_num_frames)
    for idx in range(qaa_num_frames):
        print(f"QAA snapshot time = {qaa_snapshot_times[idx]}.")
        qaa_frame = qaa_wfn[:,idx]
        qaa_nbhd_amps = qaa_frame[nbhd_idcs]
        qaa_prob = np.sum((qaa_nbhd_amps * np.conj(qaa_nbhd_amps)).real)
        qaa_prob_in_nbhd[idx] = qaa_prob
        print(f"-- QAA success probability finished.\n")

    # Compute energy ratio from t = 10 to t = 20
    ratio_times = np.arange(10.5, 20.5, 0.5)
    E_ratio_2 = np.zeros(len(ratio_times))
    counter = 0
    for t in ratio_times:
        print(f"Snapshot time = {t}.")
        tdep1 = 2 / (1e-3 + t**3)
        tdep2 = 2 * t**3
        H = (tdep1 / tdep2 * H_T) + H_U
        start = time.time()
        eigvals, eigvecs = eigsh(H, k=3, which='SA')
        end = time.time()
        print(f"-- Matrix diagonalization time = {end-start}.")
        E_ratio_2[counter] = eigvals[1] / eigvals[0]
        print(f"-- QHD energy ratio finished.\n")
        counter += 1
    E_ratio_snapshot_times = np.concatenate((snapshot_times, ratio_times), axis=0)
    E_ratio_full = np.concatenate((E_ratio, E_ratio_2), axis=0)

    # Save data
    three_phase_filename = f"./three_phase_data/{instance}_three_phase_data.npz"
    with open(three_phase_filename, 'wb') as f:
        np.savez(f,
            snapshot_times=snapshot_times,
            qhd_prob_in_nbhd=qhd_prob_in_nbhd,
            prob_spec=prob_spec,
            qaa_snapshot_times=qaa_snapshot_times,
            qaa_prob_in_nbhd=qaa_prob_in_nbhd,
            E_ratio_snapshot_times=E_ratio_snapshot_times,
            E_ratio_full=E_ratio_full
        )

    print(f"Data saved to {three_phase_filename}")
