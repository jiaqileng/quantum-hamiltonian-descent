---
layout: qp
---
In this benchmark test, we compare QHD run on the D-Wave Advantage 6.1 machine with the standard quantum adiabatic algorithm, UCSD's SNOPT, COIN-OR's IPOPT, MATLAB's `fmincon`, QCQP, and Scipy's truncated Newton method on three sets of randomly generated pentadiagonal quadratic programming problems. There are fifty problem instances per set, with sets at 50-, 60-, and 75-dimensional problems, and the metric of comparison is time to solution.

## QHD for Quadratic Programming

To begin, let's review the problem at hand.

### Quadratic programs: problem structure and generation

The goal of [quadratic programming](https://en.wikipedia.org/wiki/Quadratic_programming) is to find the unit vector that minimizes
> $$\min \frac{1}{2} x^T Q x + b^T x \ \ \text{s.t.} \ \ Q_c x \leq b_c$$.

We adjust this goal to be
> $$\min \frac{1}{2} x^T Q x + b^T x \ \ \text{s.t.} \ \ Q_c x = b_c \ \ ,\ \ 0 \leq x \leq 1$$.

For our experiments, we target problems with the box constraint, albeit without the linear equality constraint. The reason for this is that enforcing of the equality constraint via a penalty term with a fixed penalty coefficient degrades performance.

We randomly generate instances of dimension 50, 60, 75 and compare the performance of multiple methods and solvers. Due to the limited connectivity and number of qubits, the maximum dimension of the problems we are able to embed on the D-Wave QPU is roughly 75. To create the problems we uniformly sample entries for the Hessian matrices on $$[0,1]$$ with pentadiagonal structure in order to fix a maximum sparsity. This makes a minor embedding possible so that the problem can be run on the D-Wave QPU. While we do not require any equality constraints, we do enforce the box constraint $$x \in [0,1]^d$$.

### Comparisons
For each of problem sets, we compare the performance of six methods (see the Details section, 'Comparisons'): QHD on D-Wave, QAA on D-Wave, IPOPT, SNOPT, MATLAB's `fmincon` with SQP, QCQP, and a truncated Newton method.
For each instance and method, we perform 1000 trials or shots.
For the classical methods, the initial points are drawn uniformly at random from $$[0,1]^d$$.

<figure>
  <img
    src="{{ site.baseurl }}/assets/images/QPComparison.png"
    alt="QP Comparisons"
    style="display: block; margin-left: auto; margin-right: auto; width: 90%"
  />
  <figcaption>
    Experiment results for quadratic programming problems. Box plots of the time-to-solution (TTS) of selected quantum/classical solvers, gathered from four randomly generated quadratic programming benchmarks (A: 5-dimensional, B: 50-dimensional, C: 60-dimensional, D: 75-dimensional). The left and right boundaries of a box show the lower and upper quartiles of the TTS data measured by applying the corresponding solver to all instances in the benchmark, while the whiskers extend to show the rest of the TTS distribution. The median of the TTS distribution is shown as a black vertical line in the box. In each panel, the median line of the best solver extends to show the comparison with all other solvers.
  </figcaption>
</figure>
