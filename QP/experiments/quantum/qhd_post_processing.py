import numpy as np
from scipy.optimize import minimize
from scipy.optimize import Bounds
import multiprocessing
from joblib import Parallel, delayed
from os.path import join
import sys

# import data directory
sys.path.insert(1, '../../../')
from config import * 


def post_processing_qhd(benchmark_name, instance, resolution):
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
	bounds = Bounds(np.zeros(dimension), np.ones(dimension))

	def qp_fun(x):
		return 0.5 * x @ Q @ x + b @ x

	def qp_der(x):
		return Q @ x + b

	post_qhd_samples = np.zeros((numruns, dimension))
	for k in range(numruns):
		x0 = qhd_samples[k]
		result = minimize(qp_fun, x0, method='TNC', jac=qp_der, bounds=bounds,
                            options={'gtol': 1e-9, 'eps': 1e-9})
		post_qhd_samples[k] = result.x
		if k % 100 == 0:
			print(f'ID: {instance} -- The {k}-th run has completed.')

	# Save post-processed samples
	post_sample_filename = "post_" + sample_filename
	post_filename = join(instance_dir, post_sample_filename)
	with open(post_filename , 'wb') as f:
		np.save(f, post_qhd_samples)
	print(f"Benchmark: {benchmark_name}, instance: {instance}, post-processed QHD sample saved.")

	return 


if __name__ == "__main__":
	dimension = 75
	sparsity = 5
	benchmark_name = f"QP-{dimension}d-{sparsity}s"
	num_instances = 50
	resolution = 8
	num_cores = multiprocessing.cpu_count()
	print(f'Num. of cores: {num_cores}.')

	par_list = Parallel(n_jobs=num_cores)(delayed(post_processing_qhd)(benchmark_name, tid, resolution) for tid in range(num_instances))
