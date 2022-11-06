# Quantum Hamiltonian Descent

Quantum Hamiltonian Descent (QHD) is a novel framework of quantum algorithms for continuous optimization.

This is a joint work by [Jiaqi Leng](https://jiaqileng.github.io/), [Ethan Hickman](https://eth-n.github.io/), [Joseph Li](https://jli0108.github.io/), and [Xiaodi Wu](https://www.cs.umd.edu/~xwu/).


## Experiment setup

## Data
All the experiment results (raw data) are available [here](https://umd.box.com/s/vq747fvjnt8qrkbxprexhoh44n0q9m0i).

1. **NonCVX-2d**: We test two quantum algorithms (QHD and QAA) and classical algorithms (Nesterov's accelerated GD and SGD) for 22 optimization instances. Most of the instance problems are non-convex and thus considered hard for classical algorithms. In each instance folder (e.g., `/NonCVX-2d/ackley/`), the following data are provided:
  + Wave functions drawn from QHD evolution (resolution = 128, snapshot interval = 5e-1, total evolution time = 10), e.g., `/ackley_QHD128_WFN/`;
  + Wave functions drawn from QHD evolution (resolution = 256, snapshot interval = 5e-1, total evolution time = 10), e.g., `/ackley_QHD256_WFN/`;
  + Loss curve of QHD (resolution = 128, snapshot interval = 1e-3, total evolution time = 10), e.g., `/ackley_QHD128_expected_potential.npy`;
  + Loss curve of QHD (resolution = 256, snapshot interval = 1e-3, total evolution time = 10), e.g., `/ackley_QHD256_expected_potential.npy`;
  + Wave functions and loss curve of QAA (resolution = 64, total evolution time = 10), e.g., `/ackley_QAA64_T10.mat`;
  + Wave functions and loss curve of QAA (resolution = 128, total evolution time = 10), e.g., `/ackley_QAA128_T10.mat`;
  + Solution paths of Nesterov's accelerated gradient descent algorithm (num. of samples = 1000, stepsize = 1e-3, maximal iteration = 1e4, effective total evolution time = 10), e.g., `/ackley_NAGD.mat`;
  + Solution paths of stochastic gradient descent algorithm (num. of samples = 1000, stepsize = 1e-3, maximal iteration = 1e4, effective total evolution time = 10), e.g., `/ackley_SGD.mat`;
  
 2. **QP**: We test QHD and other algorithms for 160 randomly generated quadratic programming instances (dimension = 5, 50, 60, 75). These problems are non-convex (with indefinite Hessian) and have box constraints. We split the test problems into 4 benchmarks by their dimensions. In each benchmark folder (e.g., `/QP/QP-75d-5s`), the following data are provided:
  +
  
  ## Visualization of Experiment data
