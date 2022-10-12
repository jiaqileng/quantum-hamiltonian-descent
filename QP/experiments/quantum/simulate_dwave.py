import numpy as np
import random
from os.path import join
import pandas as pd
from math import comb
from scipy.sparse import csc_matrix, identity, kron
from scipy.sparse.linalg import expm_multiply
import multiprocessing
from joblib import Parallel, delayed
from itertools import product

# Please change the data directory and benchmark name
DATA_DIR = "/Users/lengjiaqi/QHD_DATA/QP"
benchmark_name = "QP-5d"
benchmark_dir = join(DATA_DIR, benchmark_name)

def get_expectation(psi, H_U):
    return np.real(psi.conjugate() @ H_U @ psi)

def get_coord_list(resolution, dimension):
    cells = [[j/resolution] for j in range(resolution+1)]
    counter = 1
    coord_list = cells
    new_list = []
    while counter < dimension:
        for p, tup in product(cells, coord_list):
            new_tup = tup.copy()
            new_tup.insert(0, p[0])
            new_list.append(new_tup)
        coord_list = new_list
        new_list = []
        counter += 1
    return coord_list


def classical_simulation(instance):
	print(f"Run classical simulation on instance {instance} from benchmark {benchmark_name}.")
	instance_dir = join(benchmark_dir, f"instance_{instance}")

	# Load instance data
	instance_filename = join(instance_dir, f"instance_{instance}.npy")
	with open(instance_filename, 'rb') as f:
		Q = np.load(f)
		b = np.load(f)
		Q_c = np.load(f)
		b_c = np.load(f)
	dimension = len(Q)
	
	# Load annealing parameters (Advantage6)
	anneal_schedule_filename = '09-1273A-A_Advantage_system6_1_annealing_schedule.xlsx'
	sheet_name = 'processor-annealing-schedule'
	advantage_df = pd.read_excel(anneal_schedule_filename, sheet_name=sheet_name)
	fraction = advantage_df['s']
	As = advantage_df['A(s) (GHz)'] #unit: GHz
	Bs = advantage_df['B(s) (GHz)'] #unit: GHz

	# D-Wave experiment parameters
	r = 8 # resolution 
	D = np.zeros((r+1,r+1))
	X = np.zeros((r+1,r+1))
	for j in range(r):
		D[j,j+1] = np.sqrt((j+1)*(r-j))
		D[j+1,j] = np.sqrt((j+1)*(r-j))
		X[j+1,j+1] = (j+1)/ r

	D = csc_matrix(D)
	X = csc_matrix(X)

	# Initial states
	u0 = np.zeros(r+1)
	for j in range(r+1):
		u0[j] = np.sqrt(comb(r,j)/2**r)
	psi0 = np.kron(u0, u0)
	for k in range(dimension-2):
		psi0 = np.kron(psi0, u0)

	# Quantum operators
	HD = csc_matrix(((r+1)**dimension, (r+1)**dimension))
	for j in range(dimension):
		Id_left = identity((r+1)**j)
		Id_right = identity((r+1)**(dimension-j-1))
		HD += kron(Id_left, kron(D, Id_right))

	HU = csc_matrix(((r+1)**dimension, (r+1)**dimension))
	for diag_id in range(dimension):
		Id_left = identity((r+1)**diag_id)
		Id_right = identity((r+1)**(dimension-diag_id-1))
		HU += 0.5 * Q[diag_id, diag_id] * kron(Id_left, kron(X**2, Id_right))
		HU += b[diag_id] * kron(Id_left, kron(X, Id_right))
	for j in range(dimension):
		for k in range(j+1, dimension):
			Id_left = identity((r+1)**j)
			Id_middle = identity((r+1)**(k-j-1))
			Id_right = identity((r+1)**(dimension-k-1))
			HU += Q[j, k] * kron(Id_left, kron(X, kron(Id_middle, kron(X, Id_right))))


	# Run simulation for time tf
	#T = [1e-3, 1e-2, 1e-1, 1] # time unit: micro-second
	T = [1]
	# sampling parameter
	numruns = 1000
	coord_list = get_coord_list(r, dimension)

	for tf_micro in T:
		tf = tf_micro * 1000 # simulation time unit: nano-second
		psi = psi0
		for j in range(len(fraction)-1):
			t1 = tf * fraction[j]
			t2 = tf * fraction[j+1]
			dt = t2 - t1
			H = -0.5*As[j]*HD + 0.5*Bs[j]*HU
			psi = expm_multiply(-1j*dt*H,psi)
			#prob_vect = np.abs(psi)**2
		expect_val = get_expectation(psi, HU)
		distribution = np.abs(psi)**2
		#distribution = prob_vect.reshape(r+1,r+1,r+1)
		print(f"instance = {instance}, tf = {tf}, expected V = {expect_val}.")

		# Generate samples from simulation data
		simulation_samples = np.zeros((numruns, dimension))
		buffer = random.choices(coord_list, weights = distribution, k = numruns)
		for i in range(numruns):
			simulation_samples[i] = buffer[i]
		'''
		# Save simulation data
		dist_filename = join(instance_path, f"{system_name}_sim_rez{r}_T{tf_micro}_distribution_{instance}.npy")
		with open(dist_filename, 'wb') as f:
			np.save(f, expect_val)
			np.save(f, distribution)
		print(f'Instance {instance}, tf = {tf_micro}, distribution saved.')
		'''

		# Save samples from distribution
		sample_filename = join(instance_dir, f"advantage6_sim_rez{r}_T{tf}_sample_{instance}.npy")
		np.save(sample_filename, simulation_samples)
		print(f'Instance {instance}, tf = {tf}, sample saved.\n')


	return

if __name__ == "__main__":
	num_cores = multiprocessing.cpu_count()
	print(f'Num. of cores: {num_cores}.')
	#num_instances = 1
	#par_list = Parallel(n_jobs=num_cores)(delayed(classical_simulation)(tid) for tid in range(num_instances))
	classical_simulation(6)