We use pseudo-spectral method to simulate QHD in two-dimensions for all the instance functions. 

## Usage
It is recommended to run the program on Linux. OpenMP is used for parallelization, so FFTW may need to be configured with `--enable-openmp`. The provided Makefile uses g++ to compile the program.

Prerequisites: [make](https://www.gnu.org/software/make/), [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page#Download), [FFTW](https://www.fftw.org/download.html), [matio](https://github.com/tbeu/matio)

We use OpenMP to parallelize the code, so FFTW should be configured with `--enable-openmp`.

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