import os
from os import mkdir, rename
from os.path import isdir, join

data_dir = "/Users/lengjiaqi/QHD_DATA/QP/QP-5d"
num_instance = 10

for instance in range(num_instance):
	instance_dir = join(data_dir, f"instance_{instance}")
	if not isdir(instance_dir):
		mkdir(instance_dir)

	file_list = [#f"instance_{instance}.npy",
	f"rand_init_{instance}.npy",
	f"gurobi_solution_{instance}.npy", 
	f"ipopt_runtime_{instance}.npy",
	f"ipopt_sample_{instance}.npy",
	f"matlab_sqp_runtime_{instance}.npy",
	f"matlab_sqp_sample_{instance}.npy",
	f"qcqp_runtime_{instance}.npy",
	f"qcqp_sample_{instance}.npy",
	f"snopt_runtime_{instance}.npy",
	f"snopt_sample_{instance}.npy",
	f"tnc_runtime_{instance}.npy",
	f"tnc_sample_{instance}.npy",
	]
	for filename in file_list:
		rename(join(data_dir,filename), join(instance_dir,filename))