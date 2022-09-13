1. Run **generate_benchmark.py** to generate benchmark set of quadratic programming instances. For each instance, the instance description, the gurobi solution (used as ground truth in the benchmark) and a sample of 1000 randomized initial guesses are generated and saved to "QHD_DATA/QP/{benchmark_name}/instance_{instance}/". Also, IPOpt is applied to the ramdomized initial guesses and the resulting solutions will be saved to the instance sub-directory as well. These data are saved in NPY format.

## Classical Experiments: 

Before running matlab-sqp and snopt on the QP problems, run **save_instances_as_mat.py** to translate the instance description and random initialization into MAT format for further usages. (These MAT files are not included in the sample data we provide on Box.)

Run **doMATLAB.m** to apply the matlab-sqp method for solving the QP problems. The solutions are saved to the instance sub-directory in MAT format. Then, one may run **save_matlab_from_mat.py** to translate the MAT file into NPY file.

Similarly, run **doSNOPT.m** to apply the snopt method for solving the QP problems. Remember to run **save_snopt_from_mat.py** to translate the MAT file into NPY file.

Run **DoAugLag.py** to apply the standard Augmented Lagrangian method for solving the QP problems. The results are directly saved to the instance sub-directory. 

## Quantum Experiments:

Prerequisites: Python 3.7.2 or greater, Amazon Braket SDK, Amazon Braket Ocean Plugin... (Good to go if you run the experiments on an AWS Braket notebook instance, use the "conda_braket" kernel)

Run the notebook **amazon-braket-QAA-QHD.ipynb** to apply the QAA and QHD algorithms for solving the QP problems. The results (samples from DWave) are saved in the instance sub-directory. 

Run the script "quantum-post-processing.py" to apply the post-processing (with the standard augmented Lagrangian method) to the quantum samples. 

