---
layout: nonconvex2d
---

We demonstrate the difference between QHD and other algorithms with interactive 3D visualizations and the three-phase diagrams (for details, see [Executive Summary](paper.html)).

## Test functions & Methodology
We gathered a set of 22 continuous two-dimensional test functions from the optimization literature (e.g., [ref1](https://www.al-roomi.org/component/content), [ref2](https://arxiv.org/abs/1308.4008), [ref3](http://www.sfu.ca/~ssurjano)).
Functions are divided into five categories based on their diversified landscapes: (1) Ridges and Valleys: a mix of functions with steep barriers and/or valleys; (2) Basin: Most of the domain is at a low objective value, so exhaustive search for would yield a small improvement in objective value. The gradient signal may be weak in most of the basin; (3) Flat: most of the domain is at a high objective value, so exhaustive search would yield a large improvement in objective value. The gradient signal may be weak, useless, or misleading over most of the domain; (4) Studded: highly non-convex functions, such that the gradient may be large in magnitude but misleading due to high local curvature. Marked by many local minima imposed on a base shape; (5) Simple: Functions solved efficiently by a classical gradient method.

For comparable performance, we normalize all objective functions. Details are available [here](https://github.com/jiaqileng/quantum-hamiltonian-descent/blob/main/NonCVX-2d/README.md). We compare QHD with the Quantum Adiabatic Algorithm (QAA), stochastic gradient descent (SGD) and Nesterov's accelerated gradient descent (NAGD) on objective functions with a variety of landscape features. QHD and QAA are numerically simulated (resolution: QHD = 1/256, QAA = 1/128).

For presentation, the surface plots are compressed to a $$64 \times 64$$ grid by taking the sum of probability over square tiles.

## Results
<figure>
  <img
    src="{{ site.baseurl }}/assets/images/ncvxSPtabT10_flat.png"
    alt="success probability"
    style="display: block; margin-left: auto; margin-right: auto; width: 90%; max-width: 1080px;">
  <figcaption>Success rate of being within r=0.1 of the global minimum in a 1x1 domain. There is a random chance of pi/100 or about 3% that the starting point is already in the radius (though the algorithms can move the point out of the radius). Classical probabilities are estimated from sample means using the same 1000 starting points. Quantum probabilities are calculated from exact full state vector simulation. Functions are sorted alphabetically by name.</figcaption>
</figure>

### Interactive Visualizations
Click the thumbnails below and have fun!