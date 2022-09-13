import numpy as np
from os.path import join

import sys
sys.path.insert(1, '../../utils')
from AugLagrangian import AugLagrangian

# Please change the data directory and benchmark name
DATA_DIR = "/Users/lengjiaqi/QHD_DATA/QP"; 
benchmark_name = "QP-75d-5s";
benchmark_dir = join(DATA_DIR, benchmark_name)
num_instances = 50

for instance in range(num_instances):
    instance_dir = join(benchmark_dir, f"instance_{instance}")
    instance_filename = join(instance_dir, f"instance_{instance}.npy")

    # Load instance problem data
    Q = np.load(instance_filename)
    b = np.load(instance_filename)
    Q_c = np.load(instance_filename)
    b_c = np.load(instance_filename)

    # Build the optimization model
    dimension = len(Q)
    lb = 0
    ub = 1
    model = AugLagrangian(Q, b, Q_c, b_c, lb, ub)

    # Specify optimization hyper-parameters
    MAX_STEPS = 1e4
    PENALTY_BASE = 10
    TOL = 1e-6
    ETA = 1e-8

    # Load random initialization
    rand_init_filename = join(instance_dir, f"rand_init_{instance}.npy")
    rand_init_samples = np.load(rand_init_filename)
    numruns = len(rand_init_samples)

    auglag_samples = np.zeros((numruns, dimension))
    for j in range(numruns):
        x0 = rand_init_samples[j]
        result = model.optimizer(x0, MAX_STEPS, PENALTY_BASE, TOL, ETA)
        xf = result["final_soln"]
        auglag_samples[k] = xf

    # Save auglag samples
    auglag_sample_filename = f"auglag_sample_{instance}.npy"
    with open(join(instance_dir, auglag_sample_filename), 'wb') as f:
        np.save(f, auglag_samples)