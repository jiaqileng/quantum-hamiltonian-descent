---
layout: nonconvex2d
---

We compare QHD with the quantum adiabatic algorithm, stochastic gradient descent and Nesterov's accelerated gradient descent on objective functions with a variety of landscape features.

## Test functions
To investigate the behavior of Quantum Hamiltonian Descent (QHD), we gathered a set of twenty-two continuous two-dimensional test functions. The functions have been normalized onto the unit square $$[0, 1]^2$$ and have been shifted so that there is a single global minimum with value 0. Functions are discretized onto a $$256 \times 256$$ grid (128 for QAA), which provides high enough resolution to avoid aliasing out local minima in the region of interest. For presentation, the surface plots are compressed to a $$64 \times 64$$ grid by taking the sum of probability over square tiles.

Functions are divided into five categories based on a subjective classification of their landscapes:
- **Ridges and Valleys:** a mix of functions with steep barriers and/or valleys
- **Basin:** Most of the domain is at a low objective value, so exhaustive search for would yield a small improvement in objective value. The gradient signal may be weak in most of the basin.
- **Flat:** most of the domain is at a high objective value, so exhaustive search would yield a large improvement in objective value. The gradient signal may be weak, useless, or misleading over most of the domain.
- **Studded:** highly non-convex functions, such that the gradient may be large in magnitude but misleading due to high local curvature. Marked by many local minima imposed on a base shape.
- **Simple:** Functions solved efficiently by a classical gradient method.

## Encoding the problems
The goal of the optimization problem is to find the function inputs that minimize the value of the objective function. To encode a test function in a quantum system, the function is used as the potential surface over the domain in which the wave function evolves. The function input is then the position in the two-dimensional potential.

The initial state of every experiment is the uniform superposition over the sites in the position basis. This choice excludes any prior knowledge about the domain from affecting the algorithm and is easy to prepare in the position/computational basis.

## Numerical simulation
Because the potential is diagonal in the position basis and the kinetic operator is diagonal in the momentum basis, we simulate the quantum dynamics by a pseudo-spectral method. This method is to apply diagonal operators sandwiched between Fourier transforms (the appropriate change of basis).


## Results
<figure>
  <img
    src="{{ site.url }}{{ site.baseurl }}/assets/images/ncvxSPtabT10.png"
    alt="success probability"
    style="display: block; margin-left: auto; margin-right: auto; width: 70%; max-width: 540px;">
  <figcaption>Table 1: Success rate of being within r=0.1 of the global minimum in a 1x1 domain. There is a random chance of pi/100 or about 3% that the starting point is already in the radius (though the algorithms can move the point out of the radius). Classical probabilities are estimated from sample means using the same 1000 starting points. Quantum probabilities are calculated from exact full state vector simulation. Functions are sorted alphabetically by name.</figcaption>
</figure>
