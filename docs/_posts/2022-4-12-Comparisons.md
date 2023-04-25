For comparison with other methods, we can examine both the path that these metrics follow and their final values.

## Algorithms

- [**Quantum Adiabatic Algorithm**](https://en.wikipedia.org/wiki/Adiabatic_quantum_computation):
  The standard implementation of the quantum adiabatic algorithm. For the 2D benchmark, the system is simulated by a Leapfrog integrator to the same simulated time as the pseudospectral method is evolved for QHD. For the QP benchmark, the annealing for QAA is carried out on the D-Wave Advantage 6.1 device. See also [https://www.cs.umd.edu/~amchilds/qa/qa.pdf#chapter.29]()

- [**(Stochastic) Gradient Descent**](https://en.wikipedia.org/wiki/Stochastic_gradient_descent):
  The functions we have are continuous and differentiable, so we have exact gradient information. We use the [stochastic differential equation formulation of SGD](https://arxiv.org/pdf/2004.06977.pdf) to numerically simulate SGD in the objective functions.

- [**Nesterov's Accelerated Gradient Descent**]:
  We use the standard form of NAGD.

- [**Interior Point OPTimizer (IPOPT)**](https://github.com/coin-or/Ipopt) (COIN-OR):
  Uses the interior point method to search within the defined bounds.

- [**Sparse Nonlinear OPTimizer (SNOPT)**](https://ccom.ucsd.edu/~optimizers/solvers/snopt/) (Center for Computational Mathematics at UCSD):
  Uses a sparse sequential quadratic programming algorithm. It is appropriate for both linear and non-linear problems as well as non-convex problems.

- [**MATLAB `fmincon`**](https://www.mathworks.com/help/optim/ug/fmincon.html):
  MATLAB's nonlinear programming solver. We use it in its sequential quadratic programming mode.

- [**Truncated Newton Method**](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-tnc.html): Offered as a method built in to SciPy as `scipy.optimize.minimize(method='TNC')`.

- [**Quadratically Constrained Quadratic Programming (QCQP)**](https://www.mathworks.com/help/optim/ug/fmincon.html):
  https://stanford.edu/~boyd/papers/qcqp.html

  <!-- - [**Augmented Lagrangian Method**](https://en.wikipedia.org/wiki/Augmented_Lagrangian_method):
    A more advanced version of the penalty method for enforcing constraints. It uses a scheduled multiplier and an adaptive multiplier together. This allows the multipliers to stay lower through the whole process, giving the method increased stability over a simple penalty method.
   -->
