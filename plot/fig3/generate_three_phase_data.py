import numpy as np
from numpy import sin, cos, exp, sqrt, pi
from scipy.sparse import diags, spdiags, identity, kron
from scipy.sparse.linalg import eigsh

import os
from os import listdir
from os.path import isfile, isdir, join
import time
import h5py
import sys

import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('../../'))
from config import *



def is_point_within_radius(point, target, radius):
    dist2 = (target[0] - point[0])**2 + (target[1] - point[1])**2
    if dist2 <= radius**2:
        return True
    return False

def grid_points_within_radius(center, radius, grid_steps):
    support = np.linspace(1/(grid_steps+1), 1 - 1/(grid_steps+1), grid_steps)
    points = []
    
    # The rows count down since the origin is now in the bottom left 
    # instead of the top left
    for c in range(grid_steps):
        for r in range(grid_steps):
            dist2 = (center[0] - support[c])**2 + (center[1] - support[-r-1])**2
            if dist2 < radius**2:
                points.append((c, grid_steps-r-1))
    return [grid_steps*c + r for (c, r) in points]


###--------------------------------------------------------------------
#Step 0: Instance Description & Setup
###--------------------------------------------------------------------
# import instance info
instance = "levy"
qhd_steps = 256
qaa_steps = 128
w = lambda v: 1 + (1/4)*(v - 1)
instance_fn = lambda x1, x2: sin(pi * w(x1))**2 + (w(x1) - 1)**2 * (1 + 10*sin(pi*w(x1) + 1)**2) + (w(x2) - 1)**2 * (1 + sin(2*pi*w(x2))**2)
instance_bounds = [-10, 10]
instance_global_min = [1, 1]

DATA_DIR = join(DATA_DIR_2D, f"{instance}")
QHD_WFN_DIR = join(DATA_DIR, f"{instance}_QHD{qhd_steps}_WFN")
QAA_WFN_DIR = join(DATA_DIR, f"{instance}_QAA{qaa_steps}_T10.mat")

# experiment setup params
max_energy_level = 10
radius = 0.1
steps = 256
stepsize = 1/(steps+1)



###--------------------------------------------------------------------
#Step 1: Construct the kinetic and potential operators
###--------------------------------------------------------------------
L = instance_bounds[1]-instance_bounds[0]
dX = L / (steps + 1)
global_min_locs = [(instance_global_min[0] - instance_bounds[0])/L,
                   (instance_global_min[1] - instance_bounds[0])/L]
meshpoints = np.linspace(instance_bounds[0]+dX, instance_bounds[1]-dX, steps)
support = np.linspace(stepsize,1-stepsize,steps)

# Kinetic operator
e = np.ones(steps)
H0 = spdiags([e, -2*e, e], [-1, 0, 1], steps, steps)
Id = identity(steps);
H_T = -0.5 / (stepsize**2) * (kron(Id,H0) + kron(H0,Id))

# Potential operator
original_V = np.empty(steps**2)
X = np.empty(steps**2)
Y = np.empty(steps**2)
for c in range(steps):
    x1 = meshpoints[c]
    for r in range(steps):
        x2 = meshpoints[-r-1]
        X[steps*c + steps-r-1] = support[c]
        Y[steps*c + steps-r-1] = support[-r-1]
        original_V[steps*c + steps-r-1] = instance_fn(x1, x2)

global_min_val = instance_fn(instance_global_min[0],instance_global_min[1])
normalized_V = (original_V - np.min(original_V[:])) / L
H_U = diags(normalized_V, 0)

###--------------------------------------------------------------------
#Step 2: Compute data used in the three-phase picture plot (QHD)
###--------------------------------------------------------------------
nbhd_idcs = grid_points_within_radius(global_min_locs, radius, steps)
nbhd_locs = np.zeros((steps * steps,1))
nbhd_locs[nbhd_idcs] = 1
nbhd_locs = np.reshape(nbhd_locs, (steps,steps), order='F')

if isdir(QHD_WFN_DIR) == False:
    print(f"The WFN data path for function {instance} is not found.")
else:
    snapshot_idx_1 = np.arange(0,10,1)
    snapshot_idx_2 = np.arange(10, 105, 5)
    snapshot_idx = np.concatenate((snapshot_idx_1,snapshot_idx_2),axis=0)
    num_frames = len(snapshot_idx)
    snapshot_times = 0.1 * snapshot_idx
    qhd_prob_in_nbhd = np.zeros(num_frames)
    prob_spec = np.zeros((max_energy_level, num_frames))
    E_ratio = np.zeros(num_frames)

    print(f"Start working on {instance} function.")

    for j in range(num_frames):
        idx = snapshot_idx[j]
        t = snapshot_times[j]
        print(f"QHD snapshot time = {t}.")
        if idx == 0:
            wfn_fname = f"psi_0.npy"
        else:
            wfn_fname = f"psi_{idx}e-01.npy"
        psi = np.load(os.path.join(QHD_WFN_DIR, wfn_fname))

        # Compute QHD success probability at time t
        qhd_prob_in_nbhd[j] = (nbhd_locs * ((psi * psi.conj()).real)).sum()
        print(f"-- QHD success probability finished.")

        # Compute QHD probability spectrum at time t
        tdep1 = 2/(1e-3 + t**3)
        tdep2 = 2*t**3
        if t == 0:
            H = tdep1 * H_T + tdep2 * H_U
        else:
            H = tdep1/tdep2 * H_T + H_U
        start = time.time()
        eigvals, eigvecs = eigsh(H, k=max_energy_level, which='SA')
        end = time.time()
        spec = psi.flatten('F') @ eigvecs
        prob_spec[:,j] = (spec * spec.conj()).real
        print(f"-- QHD probability spectrum finished, matrix diagonalization time = {end-start}.")

        # Compute QHD energy ratio at time t
        E_ratio[j] = eigvals[1]/eigvals[0]
        print(f"-- QHD energy ratio finished.\n")


###--------------------------------------------------------------------
#Step 3: Compute QAA success probability data
###--------------------------------------------------------------------
nbhd_idcs = grid_points_within_radius(global_min_locs, radius, qaa_steps)
nbhd_locs = np.zeros((qaa_steps**2,1))
nbhd_locs[nbhd_idcs] = 1
nbhd_locs = np.reshape(nbhd_locs, (qaa_steps, qaa_steps), order='F')

adb_data = {}
adb_f = h5py.File(QAA_WFN_DIR, 'r')
for k, v in adb_f.items():
    adb_data[k] = np.array(adb_f.get(k))

adb_wfn = adb_data["wfn"].view(np.complex128)
qaa_snapshot_times = adb_data["snapshot_times"][0]
adb_num_frames = len(qaa_snapshot_times)
qaa_prob_in_nbhd = np.empty(adb_num_frames)
for idx in range(adb_num_frames):
    print(f"QAA snapshot time = {qaa_snapshot_times[idx]}.")
    adb_frame = adb_wfn[:,idx]
    adb_nbhd_amps = adb_frame[nbhd_idcs]
    qaa_prob = np.sum((adb_nbhd_amps * np.conj(adb_nbhd_amps)).real)
    qaa_prob_in_nbhd[idx] = qaa_prob
    print(f"-- QAA success probability finished.\n")



###--------------------------------------------------------------------
#Step 4: Compute energy ratio from t = 10 to t = 20
###--------------------------------------------------------------------
ratio_times = np.arange(10.5, 20.5, 0.5)
E_ratio_2 = np.zeros(len(ratio_times))
counter = 0
for t in ratio_times:
    print(f"Snapshot time = {t}.")
    tdep1 = 2/(1e-3 + t**3)
    tdep2 = 2*t**3
    H = tdep1/tdep2 * H_T + H_U
    start = time.time()
    eigvals, eigvecs = eigsh(H, k=3, which='SA')
    end = time.time()
    print(f"-- Matrix diagonalization time = {end-start}.")
    E_ratio_2[counter] = eigvals[1]/eigvals[0]
    print(f"-- QHD energy ratio finished.\n")
    counter += 1
E_ratio_snapshot_times = np.concatenate((snapshot_times, ratio_times), axis=0)
E_ratio_full = np.concatenate((E_ratio, E_ratio_2), axis=0)


###--------------------------------------------------------------------
#Step 5: Save three phase data to npy
###--------------------------------------------------------------------
three_phase_filename = f"{instance}_three_phase_data.npy"
with open(three_phase_filename, 'wb') as f:
    np.save(f, snapshot_times)
    np.save(f, qhd_prob_in_nbhd)
    np.save(f, prob_spec)
    np.save(f, qaa_snapshot_times)
    np.save(f, qaa_prob_in_nbhd)
    np.save(f, E_ratio_snapshot_times)
    np.save(f, E_ratio_full)
print(f"All data saved to {three_phase_filename}.")
        


