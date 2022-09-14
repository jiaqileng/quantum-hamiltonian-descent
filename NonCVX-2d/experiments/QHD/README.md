We use pseudo-spectral method to simulate QHD in two-dimensions for all the instance functions. 


To-Do list:
1. Ethan: replace the old data files with new ones on BOX; (**Do not overwrite the data!!!**)
2. Ethan: push the new potentials256.mat (with all instances) to this repo; (replace the old one)

Let Jiaqi and Joseph know!

3. Joseph: re-run the CPP program and check it if works (make sure the output wfn data name matches the dataset Ethan uploaded); 
4. Joseph: mention the instances that require special parameters in the pseudospectral solver.
5. Jiaqi: change the qhd wave function name in the fig2 plot script. 

## Usage
It is recommended to run the program on a Linux distribution with OpenMP installed. The provided Makefile uses g++ to compile the program.

Prerequisites: [make](https://www.gnu.org/software/make/), [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page#Download), [FFTW](https://www.fftw.org/download.html), [matio](https://github.com/tbeu/matio)

Assuming the prerequisites are installed, the program can be run by simply running
```
make
./2d-pseudospec <L> <num_qubits> <T> <dt> <potentials_filename>
```
In our experiments, we set the parameters as follows:
```
./2d-pseudospec 0.5 16 500 0.005 potentials256.mat
```
For the results presented in the paper, the QHD parameters are fine-tuned for a few functions. For the Rosenbrock function, we use $\gamma$=0.05 instead of 0.01. The value of $\gamma$ is hardcoded into the program, but one may change it and run `make` again to recompile the program.

For the Holder function, we use dt=0.001 instead of 0.005.