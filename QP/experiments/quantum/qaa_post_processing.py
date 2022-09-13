import numpy as np
import io
import os
import multiprocessing
from joblib import Parallel, delayed
from os.path import join

import sys
sys.path.insert(1, '../../utils')
from AugLagrangian import AugLagrangian

# Please change the data directory and benchmark name
DATA_DIR = "/Users/lengjiaqi/QHD_DATA/QP"; 


def post_processing_qaa(benchmark_name, instance, SCHEDULE, resolution):
	print(f"Run QAA-post-processing on instance No. {instance} from {benchmark_name}.")

	benchmark_dir = join(DATA_DIR, benchmark_name)
	instance_dir = join(benchmark_dir, f"instance_{instance}")
	
	# Load instance data
	instance_filename = join(instance_dir, f"instance_{instance}.npy")
	Q = np.load(instance_filename)
	b = np.load(instance_filename)
	Q_c = np.load(instance_filename)
	b_c = np.load(instance_filename)

	# Load QAA sample data
	sample_filename = f"qaa_schedule{SCHEDULE}_rez{resolution}_sample_{instance}.npy"
	qaa_filename = join(instance_dir, sample_filename)
	qaa_samples = np.load(qaa_filename)
	numruns = len(qaa_samples)
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

	post_qaa_samples = np.zeros((numruns, dimension))
	for k in range(numruns):
		x0 = qaa_samples[k]
		result = model.optimizer(x0, MAX_STEPS, PENALTY_BASE, TOL, ETA)
		xf = result["final_soln"]
		post_qaa_samples[k] = xf
		if k % 100 == 0:
			print(f'ID: {instance} -- The {k}-th run has completed.')

	# Save post-processed samples
	post_sample_filename = "post_" + sample_filename
	post_filename = join(instance_dir, post_sample_filename)
	with open(post_filename , 'wb') as f:
        np.save(f, post_qaa_samples)

	return 


if __name__ == "__main__":
	dimension = 75
	sparsity = 5
	benchmark_name = f"QP-{dimension}d-{sparsity}s"
	SCHEDULE = "A"
	num_instances = 50
	resolution = 8
	num_cores = multiprocessing.cpu_count()
	print(f'Num. of cores: {num_cores}.')

	par_list = Parallel(n_jobs=num_cores)(delayed(post_processing_qaa)(benchmark_name, resolution, SCHEDULE, tid) for tid in range(num_instances))