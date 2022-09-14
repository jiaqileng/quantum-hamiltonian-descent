import numpy as np
import io, os
import boto3
import scipy.io
import argparse
from os.path import join, exists

# import data directory
import sys
sys.path.insert(0, '../../../')
from config import * 


parser = argparse.ArgumentParser()
parser.add_argument("benchmark_name", type=str, help="Name of benchmark")
parser.add_argument("num_instances", type=int, help="Number of instances")

args = parser.parse_args()
benchmark_name = args.benchmark_name
instances = args.num_instances

# Convert problem instances to mat files
benchmark_dir = join(DATA_DIR_QP, benchmark_name)

if not exists(benchmark_dir):
    os.mkdir(benchmark_dir)

# Load instance data from S3
for instance in range(instances):
    instance_dir = join(benchmark_dir, f"instance_{instance}")
    instance_filename = join(instance_dir, f"instance_{instance}.npy")
    
    with open(instance_filename, 'rb') as f:
        Q = np.load(f)
        b = np.load(f)
        Q_c = np.load(f)
        b_c = np.load(f)

    mat_dict = {
        'Q': Q,
        'b': b,
        'Q_c': Q_c,
        'b_c': b_c
    }
    scipy.io.savemat(file_name=join(instance_dir, f"instance_{instance}.mat"), mdict=mat_dict, oned_as='column')

    rand_init_filename = join(instance_dir, f"rand_init_{instance}.npy")
    rand_init = np.load(rand_init_filename)

    mat_dict = {
        'rand_init': rand_init
    }
    scipy.io.savemat(file_name=join(instance_dir, f"rand_init_{instance}.mat"), mdict=mat_dict, oned_as='column')
