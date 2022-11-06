We use pseudo-spectral method to simulate QHD in two-dimensions for all the instance functions. 

## Usage
It is recommended to run the program on Linux. OpenMP is used for parallelization, so FFTW may need to be configured with `--enable-openmp`. The provided Makefile uses g++ to compile the program.

Prerequisites: [make](https://www.gnu.org/software/make/), [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page#Download), [FFTW](https://www.fftw.org/download.html), [matio](https://github.com/tbeu/matio)

We use OpenMP to parallelize the code, so FFTW should be configured with `--enable-openmp`.

Assuming the prerequisites are installed, the program can be run by simply running
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

The MATLAB script `savePotentials.m` can be used in conjunction with `experimentSetup.m` in the `NonCVX-2d/util` directory to generate potential files for any resolution. The variable `num_cells` should be set to 128 or 256.