import numpy as np
import sys
from scipy.optimize import minimize
from scipy.optimize import Bounds
from os.path import join

# import data directory
sys.path.insert(1, '../../../')
from config import * 

# specify benchmark
dimension = 75
num_instances = 1
benchmark_name = f"QP-{dimension}d-5s"
benchmark_dir = join(DATA_DIR_QP, benchmark_name)



for instance in range(num_instances):
    instance_dir = join(benchmark_dir, f"instance_{instance}")
    instance_filename = join(instance_dir, f"instance_{instance}.npy")

    # Load instance problem data
    with open(instance_filename, 'rb') as f:
        Q = np.load(f)
        b = np.load(f)
        Q_c = np.load(f)
        b_c = np.load(f)

    # Build the optimization model
    dimension = len(Q)
    bounds = Bounds(np.zeros(dimension), np.ones(dimension))

    def qp_fun(x):
        return 0.5 * x @ Q @ x + b @ x

    def qp_der(x):
        return Q @ x + b

    # Load random initializations
    rand_init_filename = join(instance_dir, f"rand_init_{instance}.npy")
    rand_init_samples = np.load(rand_init_filename)
    numruns = len(rand_init_samples)

    # Run TNC solver
    tnc_samples = np.zeros((numruns, dimension))
    for j in range(numruns):
        x0 = rand_init_samples[j]
        result = minimize(qp_fun, x0, method='TNC', jac=qp_der, bounds=bounds,
                            options={'gtol': 1e-9, 'eps': 1e-9})
        tnc_samples[j] = result.x

    # Save TNC sample
    tnc_sample_filename = f"tnc_sample_{instance}.npy"
    with open(join(instance_dir, tnc_sample_filename), 'wb') as f:
        np.save(f, tnc_samples)
    print(f"Benchmark: {benchmark_name}, instance: {instance}, TNC sample saved.")