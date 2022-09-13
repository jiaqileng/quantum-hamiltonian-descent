import numpy as np
from numpy import sin, cos, exp, sqrt, pi
from scipy.sparse import diags, spdiags, identity, kron
from scipy.sparse.linalg import eigsh

import os
from os import listdir
from os.path import isfile, isdir, join

import time


# Please change the data directory and benchmark name
DATA_DIR = "/Users/lengjiaqi/QHD_DATA/NonCVX-2d"



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
instance = "styblinski_tang"
instance_fn = lambda x1,x2: 1/78 * 0.5 * (x1**4 - 16*x1**2 + 5*x1 + x2**4 - 16*x2**2 + 5*x2)
instance_bounds = [-5, 5]
instance_global_min = [-2.9035, -2.9035]
gamma = 0.01
QHD_WFN_DIR = join(DATA_DIR, f"{instance}/{instance}_QHD_WFN")

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
#Step 2: Compute energy ratios
#-----------------------------------------------------
time_range1 = np.linspace(0,105,50)
time_range2 = np.arange(105.1,105.3,step=0.01)
time_range3 = np.linspace(105.35,200,50)
time_range = np.concatenate((time_range1, time_range2, time_range3), axis=None)
E_ratio = np.zeros(len(time_range))

for j in range(len(time_range)):
    t = time_range[j]
    tdep1 = 1/(1 + gamma*t**2)**2
    tdep2 = 1
    H = tdep1 * H_T + tdep2 * H_U

    start = time.time()
    eigvals, eigvecs = eigsh(H, k=2, which='SA')
    E_ratio[j] = eigvals[1]/eigvals[0]
    end = time.time()
    print(f"t = {t}, time elapsed = {end-start}.\n")

#-----------------------------------------------------
#Step 3: Save energy ratio data from computation
#-----------------------------------------------------
energy_ratio_filename = f"{instance}_energy_ratio.npy"
if not isfile(energy_ratio_filename):
    with open(energy_ratio_filename, 'wb') as f:
        np.save(f, time_range)
        np.save(f, E_ratio)