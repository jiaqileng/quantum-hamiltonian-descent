import numpy as np
import scipy
import gurobipy as gp

def construct_sparse(bandwidth : int, dimension : int):
    diags = [(2 * np.random.rand(dimension - i) - 1) for i in range(bandwidth + 1)]
    offsets = list(range(bandwidth + 1))
    A = scipy.sparse.diags(diags, offsets).toarray()
    return (A + A.T) / 2

# Generates equality constraints randomly such that feasible region is nonempty
def random_constraints(dimension, num_constraints, bandwidth):
    x = np.random.rand(dimension)
    diags = [(np.random.uniform(low=0, high=1, size=dimension - np.abs(i))) for i in range(bandwidth+1)]
    offsets = list(range(bandwidth+1))
    Q_c = scipy.sparse.diags(diags, offsets).toarray()
    Q_c = Q_c[:num_constraints]
    b_c = Q_c @ x
    return Q_c, b_c

def solve_gurobi(Q, b, Q_c, b_c):
    # Create a new model
    m = gp.Model()
    # Create variables
    x = m.addMVar(b.size, vtype=gp.GRB.CONTINUOUS, lb=0, ub=1)

    # Set objective function
    m.setObjective((x @ Q @ x) * 0.5 + b @ x, gp.GRB.MINIMIZE)
    
    # Add constraints
    m.addConstrs((b_c[i] <= Q_c[i] @ x for i in range(b_c.size)), name='c')
    m.addConstrs((Q_c[i] @ x <= b_c[i] for i in range(b_c.size)), name='c')
    
    m.update()

    # Need this to solve non-convex QP
    m.setParam('NonConvex', 2)

    # Can set to 1 for dual simplex or 2 for barrier
    m.setParam('Method', 1)
    m.setParam('OutputFlag', 0)
    
    m.setParam('Threads', 1)

    # Solve it!
    m.optimize()

    return m, x