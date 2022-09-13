import numpy as np
import random
import io
import os
from os.path import join
import pandas as pd
from math import comb
from scipy.sparse import csc_matrix, identity, kron
from scipy.sparse.linalg import expm_multiply
import multiprocessing
from joblib import Parallel, delayed


# Please change the data directory and benchmark name
DATA_DIR = "/Users/lengjiaqi/QHD_DATA/QP";
benchmark_name = "QP-3d"
benchmark_dir = join(DATA_DIR, benchmark_name)

def get_expectation(psi, H_U):
    return np.real(psi.conjugate() @ H_U @ psi)


def classical_simulation(instance):
	print(f"Run classical simulation on instance {instance} from benchmark {benchmark_name}.")
	instance_path = join(benchmark_dir, f"instance_{instance}")
	
	# Load annealing parameters
	system_name = 'advantage6'
	anneal_schedule_filename = '09-1273A-A_Advantage_system6_1_annealing_schedule.xlsx'
	sheet_name = 'processor-annealing-schedule'
	advantage_df = pd.read_excel(anneal_schedule_filename, sheet_name=sheet_name)
	fraction = advantage_df['s']
	As = advantage_df['A(s) (GHz)'] #unit: GHz
	Bs = advantage_df['B(s) (GHz)'] #unit: GHz

	# D-Wave experiment parameters
	r = 8 # resolution 
	Id = identity(r+1)
	D = np.zeros((r+1,r+1))
	X = np.zeros((r+1,r+1))
	for j in range(r):
	    D[j,j+1] = np.sqrt((j+1)*(r-j))
	    D[j+1,j] = np.sqrt((j+1)*(r-j))
	    X[j+1,j+1] = (j+1)/ r

	D = csc_matrix(D)
	X = csc_matrix(X)

	# Quantum operators (3-dim)
	HD = kron(kron(D,Id),Id) + kron(kron(Id,D),Id) + kron(kron(Id,Id),D)

	# Initial state
	u0 = np.zeros(r+1)
	for j in range(r+1):
	    u0[j] = np.sqrt(comb(r,j)/2**r)
	psi0 = np.kron(np.kron(u0,u0),u0)
	
	# Load instance data from S3
	instance_filename = join(instance_path, f"instance_{instance}.npy")
	Q = np.load(instance_filename)
	b = np.load(instance_filename)
	dimension = len(Q)

	# Build the potential operator (d=3)
	HU = 0.5*(Q[0,0]*kron(kron(X**2,Id),Id) + Q[1,1]*kron(kron(Id,X**2),Id) + Q[2,2]*kron(kron(Id,Id),X**2))
	HU += Q[0,1]*kron(kron(X,X),Id) + Q[0,2]*kron(kron(X,Id),X) + Q[1,2]*kron(kron(Id,X),X)
	HU += b[0]*kron(kron(X,Id),Id) + b[1]*kron(kron(Id,X),Id) + b[2]*kron(kron(Id,Id),X)

	# Run simulation for time tf
	T = [1e-3, 1e-2, 1e-1, 1, 2, 5, 10, 20, 50] # time unit: micro-second

	# sampling parameter
	numruns = 1000
	cells = np.linspace(0, 1, r+1)
	coord_list = []
	for i in range(r+1):
		for j in range(r+1):
			for k in range(r+1):
				coord_list.append([cells[i],cells[j],cells[k]])

	for tf_micro in T:
		tf = tf_micro * 1000 # simulation time unit: nano-second
		t0 = 0
		psi = psi0
		for j in range(len(fraction)-1):
			t1 = tf * fraction[j]
			t2 = tf * fraction[j+1]
			dt = t2 - t1
			H = -0.5*As[j]*HD + 0.5*Bs[j]*HU
			psi = expm_multiply(-1j*dt*H,psi)
			prob_vect = np.abs(psi)**2
		expect_val = get_expectation(psi, HU)
		distribution = prob_vect.reshape(r+1,r+1,r+1)
		print(f"instance = {instance}, tf = {tf}, expected V = {expect_val}.")

		# Generate samples from simulation data
		simulation_samples = np.zeros((numruns, dimension))
		buffer = random.choices(coord_list, weights = distribution.flatten(), k = numruns)
		for i in range(numruns):
			simulation_samples[i] = buffer[i]

		# Save simulation data
		dist_filename = join(instance_path, f"{system_name}_sim_rez{r}_T{tf_micro}_distribution_{instance}.npy")
		with open(dist_filename, 'wb') as f:
			np.save(f, expect_val)
			np.save(f, distribution)
		print(f'Instance {instance}, tf = {tf_micro}, distribution saved.')

		# Save ssamples from distribution
		sample_filename = join(instance_path, f"advantage6_sim_rez{r}_T{tf}_sample_{instance}.npy")
		np.save(sample_filename, simulation_samples)
		print(f'Instance {instance}, tf = {tf}, sample saved.\n')


	return

if __name__ == "__main__":
	num_cores = multiprocessing.cpu_count()
	print(f'Num. of cores: {num_cores}.')
	num_instances = 10
	par_list = Parallel(n_jobs=num_cores)(delayed(classical_simulation)(tid) for tid in range(num_instances))