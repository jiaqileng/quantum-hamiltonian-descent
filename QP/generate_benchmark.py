import numpy as np
import os
from os.path import join, exists

import sys
sys.path.insert(1, '/utils')
from generate_benchmark_utils import construct_sparse, random_constraints, solve_gurobi

# Ipopt
import cyipopt
import argparse

# Please change the data directory and benchmark name
DATA_DIR = "/Users/lengjiaqi/QHD_DATA/QP"

class QuadraticProgram():
    # This class describes the quadratic programming problem:
    # min_x 0.5 * x^T * Q * x + b^T x, 
    # subject to lb <= x <= ub, cl <= Q_c * x <= cu
    def __init__(self, dim, num_cons, Q, b, Q_c, lb, ub, cl, cu):
        self.dim = dim # num. of continuous variables
        self.num_cons = num_cons # num. of (inequality) constraints
        self.Q = Q 
        self.b = b 
        self.Q_c = Q_c
        self.lb = lb
        self.ub = ub
        self.cl = cl
        self.cu = cu
    def objective(self, x):
        """Returns the scalar value of the objective given x."""
        
        return (x @ self.Q @ x) / 2 + self.b @ x
    def gradient(self, x):
        """Returns the gradient of the objective with respect to x."""
        return self.Q @ x + self.b
    def constraints(self, x):
        """Returns the constraints."""
        return np.array([self.Q_c[i] @ x for i in range(self.num_cons)])
    def jacobian(self, x):
        """Returns the Jacobian of the constraints with respect to x."""
        return np.array([self.Q_c[i] for i in range(self.num_cons)]).flatten()
    def hessianstructure(self):
        
        return np.nonzero(np.ones(shape=(self.dim, self.dim)))
    def hessian(self, x, lagrange, obj_factor):
        """Returns the non-zero values of the Hessian."""
        H = obj_factor * self.Q
        for i in range(self.num_cons):
            H += lagrange[i] * self.Q_c[i]
        #row, col = self.hessianstructure()
        return H.flatten()
    def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu,
                     d_norm, regularization_size, alpha_du, alpha_pr,
                     ls_trials):
        """Prints information at every Ipopt iteration."""

        #print(f"Objective value at iteration #{iter_count} is {obj_value}")


parser = argparse.ArgumentParser()
parser.add_argument("dimension", nargs=1, type=int, help="Number of decision variables")
parser.add_argument("bandwidth", nargs=1, type=int, help="Bandwidth for band matrix Hessian")
parser.add_argument("num_constraints", nargs=1, type=int, help="Number of constraints")
args = parser.parse_args()
dimension = args.dimension[0]
bandwidth = args.bandwidth[0]
num_constraints = args.num_constraints[0]

numruns = 1000
sparsity = min(2 * bandwidth + 1, dimension)

count = 0
while count<50:
    benchmark_name = f"QP-{dimension}d-{sparsity}s"
    instance_path = join(DATA_DIR, f"{benchmark_name}/instance_{count}")

    if not exists(instance_path):
        os.mkdir(instance_path)
    
    # Generate problem (band matrices for objective and constraints)
    Q = construct_sparse(bandwidth, dimension)
    b = np.random.rand(dimension)
    Q_c, b_c = random_constraints(dimension, num_constraints, bandwidth)
    
    # Solve problem with Gurobi to get ground truth
    m, x = solve_gurobi(Q, b, Q_c, b_c)
    x = x.X
    gurobi_solution = x

    ground_loss = (x @ Q @ x) / 2 + b @ x
    
    results = np.zeros(numruns)

    num_cons = b_c.size
    lb = np.zeros(dimension)
    ub = np.ones(dimension)
    cl = b_c
    cu = b_c
    ipopt_instance = QuadraticProgram(dim=dimension, num_cons=num_cons, Q=Q, b=b, Q_c=Q_c, lb=lb, ub=ub, cl=cl, cu=cu)

    nlp = cyipopt.Problem(
       n=ipopt_instance.dim,
       m=ipopt_instance.num_cons,
       problem_obj=ipopt_instance,
       lb=ipopt_instance.lb,
       ub=ipopt_instance.ub,
       cl=ipopt_instance.cl,
       cu=ipopt_instance.cu,
    )
    nlp.add_option('print_level', 0)
    # Generate random initializations
    rand_init_x = np.random.rand(numruns, dimension)

    # Collect samples
    ipopt_samples = np.zeros(shape=(numruns, dimension))
    for i in range(numruns):
        x, _ = nlp.solve(rand_init_x[i])
        ipopt_samples[i] = x
        results[i] = (x @ Q @ x) / 2 + b @ x
    
    # Save gurobi solution
    filename = f"gurobi_solution_{count}.npy"
    with open(join(instance_path, filename), 'wb') as f:
        np.save(f, gurobi_solution)

    # Save random initializations
    filename = f"rand_init_{count}.npy"
    with open(join(instance_path, filename), 'wb') as f:
        np.save(f, rand_init_x)

    # Save IPOpt samples
    filename = f"ipopt_sample_{count}.npy"
    with open(join(instance_path, filename), 'wb') as f:
        np.save(f, ipopt_samples)

    # Save problem instance
    filename = f"instance_{count}.npy"
    with open(join(instance_path, filename), 'wb') as f:
        np.save(f, Q)
        np.save(f, b)
        np.save(f, Q_c)
        np.save(f, b_c)

    print(f"Instance {count}.")
    count += 1