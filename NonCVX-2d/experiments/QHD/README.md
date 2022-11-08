# Pseudospectral Solver for Simulating QHD
We use pseudo-spectral method to simulate QHD in two-dimensions for all the instance functions. 

## Usage
### Prerequisites
It is recommended to run the program on Linux, for which the program has been developed. OpenMP is used to parallelize the code, so FFTW should be configured with `--enable-openmp`. The provided Makefile uses g++ to compile the program.

Prerequisites: [make](https://www.gnu.org/software/make/), [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page#Download), [FFTW](https://www.fftw.org/download.html), [matio](https://github.com/tbeu/matio)

The MATLAB script `savePotentials.m` can be used in conjunction with `experimentSetup.m` in the `NonCVX-2d/util` directory to generate potential files for any resolution. The files `potentials128.mat` and `potentials256.mat` are provided for convenience. In `savePotentials.m`, the variable `num_cells` should be set to 128 or 256, or another resolution if the appropriate potentials file is available.

### Saving the wavefunctions
One can modify `2d-pseudospec.cpp` to adjust how often the wave function is saved.
For the first `NUMBER_OF_ITERATIONS_A` iterations, the wavefunction is saved every `CAPTURE_FRAME_EVERY_A` iterations.
Afterwards, the wavefunction is saved every `CAPTURE_FRAME_EVERY_B` iterations.

By default, the wavefunction files are named with the time truncated to the tenths place, i.e. `psi_{k}e-01.npy`, where `k` is an integer. This means that for `dt=0.001`, `CAPTURE_FRAME_EVERY_A` and `CAPTURE_FRAME_EVERY_B` should be set to a multiple of 100 so that the filenames are accurate.

### Compiling and running the program
Assuming the prerequisites are installed, the program can be compiled and run using the following commands:
```
make
./2d-pseudospec <L> <num_cells> <T> <dt> <potentials_filename>
```
- `L` is the size of the box centered at the origin. We use `L=0.5` for our experiments.
- `num_cells` is the number of points in each dimension due to spatial discretization. In our experiments, we use either `num_cells=128` or `num_cells=256`.
- `T` is the evolution time. In our experiments, `T=10`.
- `dt` is the time step for each iteration due to time discretization. In our experiments, `dt=0.001`.

In our experiments with a higher resolution of 256, we set the parameters as follows:
```
./2d-pseudospec 0.5 256 10 0.001 potentials256.mat
```
One can also use `potentials128.mat` to run the program with lower resolution in the spatial discretization, as folows:
```
./2d-pseudospec 0.5 128 10 0.001 potentials128.mat
```

