import numpy as np

import sys
sys.path.insert(1, '/utils')
from generate_benchmark_utils import construct_sparse, random_constraints, solve_gurobi

num_constraints = 0
runs_per_dim = 100
dim_high = [201, 201, 201, 96]
dim_step = 5
sparsities = [5, 15, 25, 45]

for k, sparsity in enumerate(sparsities):
    bandwidth = int((sparsity - 1) / 2)

    dimensions = np.arange(sparsity, dim_high[k], dim_step)
    gurobi_runtimes = np.zeros((len(dimensions), runs_per_dim))

    for i, dimension in enumerate(dimensions):
        for j in range(runs_per_dim):
            Q = construct_sparse(bandwidth, dimension)
            b = np.random.rand(dimension)
            Q_c, b_c = random_constraints(dimension, num_constraints, bandwidth)
            m, x = solve_gurobi(Q, b, Q_c, b_c)
            gurobi_runtimes[i, j] = m.Runtime

        avg_runtime = np.average(gurobi_runtimes[i,:])

        print(f"Dim {dimension}, bandwidth {bandwidth}, runtime: {avg_runtime}")

    np.save(f"gurobi_runtime_sparsity_{sparsity}.npy", gurobi_runtimes)