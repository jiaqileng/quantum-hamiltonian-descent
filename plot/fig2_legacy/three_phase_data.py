import numpy as np
from numpy import sin, cos, exp, sqrt, pi
from scipy.sparse import diags, spdiags, identity, kron
from scipy.sparse.linalg import eigsh

import os
from os import listdir
from os.path import isfile, isdir, join
import time
import sys


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



# import instance info
instance = "levy"
DATA_DIR = join(DATA_DIR_2D, f"{instance}")
QHD_WFN_DIR = join(DATA_DIR, f"{instance}_QHD256_WFN")
w = lambda v: 1 + (1/4)*(v - 1)
levy_fn = lambda x1, x2: sin(pi * w(x1))**2 + (w(x1) - 1)**2 * (1 + 10*sin(pi*w(x1) + 1)**2) + (w(x2) - 1)**2 * (1 + sin(2*pi*w(x2))**2)
instance_bounds = [-10, 10]
instance_global_min = [1, 1]

# experiment setup params
max_energy_level = 10
radius = 0.1
steps = 256
stepsize = 1/(steps+1)


#-----------------------------------------------------
#Step 1: Construct the kinetic and potential operators
#-----------------------------------------------------
L = instance_bounds[1]-instance_bounds[0]
dX = L / (steps + 1)
global_min_locs = [(instance_global_min[0] - instance_bounds[0])/L,
                   (instance_global_min[1] - instance_bounds[0])/L]
meshpoints = np.linspace(instance_bounds[0]+dX, instance_bounds[1]-dX, steps)
support = np.linspace(stepsize,1-stepsize,steps)

# Kinetic operator
e = np.ones(steps)
B = spdiags([e, -4*e, e], [-1, 0, 1], steps, steps)
A = spdiags([e, 0*e, e], [-1, 0, 1], steps, steps)
Id = identity(steps);
H_T = -0.5 / (stepsize**2) * (kron(Id,B) + kron(A,Id))

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
normalized_V = (original_V - global_min_val) / L
H_U = diags(normalized_V, 0)

#-----------------------------------------------------
#Step 2: Retrieve QHD wave functions & probability
#-----------------------------------------------------
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
    qhd_prob_in_nbhd = []
    for idx in snapshot_idx:
        if idx == 0:
            wfn_fname = f"psi_0.npy"
        else:
            wfn_fname = f"psi_{idx}e-01.npy"
        psi = np.load(os.path.join(QHD_WFN_DIR, wfn_fname))
        prob_nbhd = (nbhd_locs * ((psi * psi.conj()).real)).sum()
        qhd_prob_in_nbhd.append(prob_nbhd)

#-----------------------------------------------------
#Step 3: Spectrum decomposition
#-----------------------------------------------------
prob_spec = np.zeros((max_energy_level, num_frames))
for j in range(num_frames):
    idx = snapshot_idx[j]
    if idx == 0:
        wfn_fname = f"psi_0.npy"
    else:
        wfn_fname = f"psi_{idx}e-01.npy"
    psi = np.load(os.path.join(QHD_WFN_DIR, wfn_fname)).flatten('F')
    t = snapshot_times[j]
    tdep1 = 2/(1e-3 + t**3)
    tdep2 = 2*t**3
    H = tdep1 * H_T + tdep2 * H_U

    start = time.time()
    eigvals, eigvecs = eigsh(H, k=max_energy_level, which='SA')
    spec = psi @ eigvecs
    prob_spec[:,j] = (spec * spec.conj()).real
    end = time.time()
    print(f"t = {t}, time elapsed = {end-start}.\n")


#-----------------------------------------------------
#Step 4: Save three phase data from computation
#-----------------------------------------------------
three_phase_filename = f"{instance}_three_phase.npy"
if not isfile(three_phase_filename):
	with open(three_phase_filename, 'wb') as f:
		np.save(f, times)
		np.save(f, prob_spec)
		np.save(f, qhd_prob_in_nbhd)