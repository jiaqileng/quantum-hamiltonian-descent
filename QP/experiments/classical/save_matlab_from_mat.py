import numpy as np
import io, os
import scipy.io
import argparse
from os.path import join

# import data directory
import sys
sys.path.insert(0, '../../../')
from config import * 



benchmark_name = "QP-75d-5s";
benchmark_dir = join(DATA_DIR_QP, benchmark_name)
num_instances = 50

for instance in range(num_instances):
    instance_dir = join(benchmark_dir, f"instance_{instance}")
    
    # Load data from mat file
    matlab_data = scipy.io.loadmat(file_name=join(instance_dir, f"matlab_sqp_sample_{instance}.mat"))
    matlab_sample = matlab_data['matlab_sample']

    # Save final data to npy
    matlab_filename = f"matlab_sqp_sample_{instance}.npy"
    with open(join(instance_dir, matlab_filename), 'wb') as f:
        np.save(f, matlab_sample)