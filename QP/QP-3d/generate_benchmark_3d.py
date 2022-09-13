import numpy as np
import os
from os.path import join, exists

# Please change the data directory and benchmark name
DATA_DIR = "/Users/lengjiaqi/QHD_DATA/QP";
benchmark_name = "QP-3d"
benchmark_dir = join(DATA_DIR, benchmark_name)

 # Generate 3D QP instances
d = 3
num_test_instances = 10

for instance in range(num_test_instances):
    instance_path = join(benchmark_dir, f"instance_{instance}")

    if not exists(instance_path):
        os.mkdir(instance_path)

	var = 10
    Q = var * np.random.randn(d,d)
    Q = 0.5*(Q + Q.T)
    b = var * np.random.randn(d)
    Q_c = np.zeros((0, d))
    b_c = np.array([])

    # Solve with Gurobi

    # Save new instance data 
    instance_filename = join(instance_path, f"instance_{instance}.npy")
    with open(instance_filename, 'wb') as f:
        np.save(instance_filename, Q)
        np.save(instance_filename, b)
        np.save(instance_filename, Q_c)
        np.save(instance_filename, b_c)
    
    print(f"Instance {instance} has been saved.")