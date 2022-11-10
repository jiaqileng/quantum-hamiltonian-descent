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
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

sys.path.append(os.path.abspath('../../'))
from config import *



###-----------------------------------------------------
#Step 0: Instance Description & Setup
###-----------------------------------------------------
# import instance info
instance = "levy"
qhd_steps = 256
w = lambda v: 1 + (1/4)*(v - 1)
instance_fn = lambda x1, x2: sin(pi * w(x1))**2 + (w(x1) - 1)**2 * (1 + 10*sin(pi*w(x1) + 1)**2) + (w(x2) - 1)**2 * (1 + sin(2*pi*w(x2))**2)
instance_bounds = [-10, 10]
instance_global_min = [1, 1]

DATA_DIR = join(DATA_DIR_2D, f"{instance}")
QHD_WFN_DIR = join(DATA_DIR, f"{instance}_QHD{qhd_steps}_WFN")

# experiment setup params
max_energy_level = 3
radius = 0.1
steps = 256
stepsize = 1/(steps+1)



###-----------------------------------------------------
#Step 1: Construct the kinetic and potential operators
###-----------------------------------------------------
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



###-----------------------------------------------------
#Step 2: Compute eigenstates of H
###-----------------------------------------------------
state_0 = np.zeros((max_energy_level, steps**2))
state_10 = np.zeros((max_energy_level, steps**2))

# t = 0
t = 0
H = H_T
start = time.time()
eigvals, eigvecs = eigsh(H, k=max_energy_level, which='SA')
end = time.time()
print(f"t = {t}, matrix diagonalization time = {end-start}.")
for k in range(max_energy_level):
    state_0[k,:] = eigvecs[:,k]

# t = 10
t = 10
tdep1 = 2/(1e-3 + t**3)
tdep2 = 2*t**3
H = tdep1/tdep2 * H_T + H_U
start = time.time()
eigvals, eigvecs = eigsh(H, k=max_energy_level, which='SA')
end = time.time()
print(f"t = {t}, matrix diagonalization time = {end-start}.")   
for k in range(max_energy_level):
    state_10[k,:] = eigvecs[:,k]



###-----------------------------------------------------
#Step 3: Plot & Save
###-----------------------------------------------------
TITLE_FONT = 14
CMAP = 'RdBu'
state_identifier = ['(0,0)','(0,1)','(1,0)']

sns.set_theme()
fig = plt.figure(figsize=(12,1.9))
gs = fig.add_gridspec(1, 6)

ax = fig.add_subplot(gs[0,0:3])
ax.set_title('QHD first 3 eigen-states at t = 0', fontsize=TITLE_FONT)
ax.set_facecolor('white')
ax.set_xticks([])
ax.set_yticks([])
for col in range(3):
    #t = times[col]
    state = state_0[col,:].reshape((steps,steps))
    if k == 0:
        state = np.abs(state)
    ax = fig.add_subplot(gs[0,col])
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(state_identifier[col])
    im = ax.imshow(state, 
              origin='lower',
              cmap=CMAP,
              vmin=-0.01,
              vmax=0.01,
            )
    #plt.colorbar(im, cmap=CMAP, ticks=[-0.01,0,0.01], fraction=0.046, pad=0.04, shrink=0.8)


ax = fig.add_subplot(gs[0,3:6])
ax.set_title('QHD first 3 eigen-states at t = 10', fontsize=TITLE_FONT)
ax.set_facecolor('white')
ax.set_xticks([])
ax.set_yticks([])
for col in range(3):
    state = state_10[col,:].reshape((steps,steps))
    if k == 0:
        state = np.abs(state)
    ax = fig.add_subplot(gs[0,col+3])
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(state_identifier[col])
    im = ax.imshow(state, 
              origin='lower',
              cmap=CMAP,
              vmin=-0.01,
              vmax=0.01,
            )
    #if col == 2:
    #    plt.colorbar(im, cmap=CMAP, ticks=[-0.01,0,0.01], fraction=0.5, pad=0.1)
        
#plt.show()
plt.savefig('figures/lowE.png')
plt.savefig('figures/lowE.eps',dpi=300)