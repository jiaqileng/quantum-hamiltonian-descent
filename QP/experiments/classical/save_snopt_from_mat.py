import numpy as np
import io, os
import boto3
import scipy.io
import argparse
from os.path import join

# Please change the data directory and benchmark name
DATA_DIR = "/Users/lengjiaqi/QHD_DATA/QP"
benchmark_name = "QP-75d-5s";
benchmark_dir = join(DATA_DIR, benchmark_name)
num_instances = 50

for instance in range(num_instances):
    instance_dir = join(benchmark_dir, f"instance_{instance}")

    # Load data from mat file
    snopt_data = scipy.io.loadmat(file_name=join(instance_dir, f"snopt_sample_{instance}.mat"))
    snopt_sample = snopt_data['snopt_sample']

    # Save final data to npy
    snopt_filename = f"snopt_sample_{instance}.npy"
    with open(join(instance_dir, snopt_filename), 'wb') as f:
        np.save(f, snopt_sample)