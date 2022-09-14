import numpy as np
import io
import os
import multiprocessing
from joblib import Parallel, delayed
from os.path import join

import sys
sys.path.insert(0, '../')
from AugLagrangian import AugLagrangian

# import data directory
sys.path.insert(1, '../../../')
from config import * 


def post_processing(benchmark_name, instance, resolution):
	print(f"Run QHD-post-processing on instance No. {instance} from {benchmark_name}.")

	benchmark_dir = join(DATA_DIR_QP, benchmark_name)
	instance_dir = join(benchmark_dir, f"instance_{instance}")
	
	# Load instance data
	instance_filename = join(instance_dir, f"instance_{instance}.npy")
	with open(instance_filename, 'rb') as f:
		Q = np.load(f)
		b = np.load(f)
		Q_c = np.load(f)
		b_c = np.load(f)

	# Load QHD sample data
	sample_filename = f"advantage6_qhd_rez{resolution}_sample_{instance}.npy"
	qhd_filename = join(instance_dir, sample_filename)
	qhd_samples = np.load(qhd_filename)
	numruns = len(qhd_samples)
	print(f'ID: {instance} -- Number of runs: {numruns}.')

	# Build the post-processing model
	dimension = len(Q)
	lb = 0
	ub = 1
	model = AugLagrangian(Q, b, Q_c, b_c, lb, ub)

	# Specify optimization hyper-parameters
	MAX_STEPS = 1e4
	PENALTY_BASE = 10
	TOL = 1e-6
	ETA = 1e-8

	post_qhd_samples = np.zeros((numruns, dimension))
	for k in range(numruns):
		x0 = qhd_samples[k]
		result = model.optimizer(x0, MAX_STEPS, PENALTY_BASE, TOL, ETA)
		xf = result["final_soln"]
		post_qhd_samples[k] = xf
		if k % 100 == 0:
			print(f'ID: {instance} -- The {k}-th run has completed.')

	# Save post-processed samples
	post_sample_filename = "post_" + sample_filename
	post_filename = join(instance_dir, post_sample_filename)
	with open(post_filename , 'wb') as f:
		np.save(f, post_qhd_samples)

	return 


if __name__ == "__main__":
	dimension = 75
	sparsity = 5
	benchmark_name = f"QP-{dimension}d-{sparsity}s"
	num_instances = 50
	resolution = 8
	num_cores = multiprocessing.cpu_count()
	print(f'Num. of cores: {num_cores}.')

	par_list = Parallel(n_jobs=num_cores)(delayed(post_processing)(benchmark_name, tid, resolution) for tid in range(num_instances))
