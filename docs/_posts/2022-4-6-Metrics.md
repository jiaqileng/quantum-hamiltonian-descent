---
author: ethan
layout: post
---
We use standard metrics for benchmarking QHD against the other optimization algorithms: **success probability** and **time-to-solutions (TTS)**.

## Success Probability
Existing methods for non-convex optimization have little or no theoretical guarantee of convergence to global extrema. Of course, many optimization applications are most interested in finding a global minimizer. This makes empirical study of an algorithm's success rate on a variety of problems an interesting metric for optimization benchmarking.

For the nonconvex 2D test functions, we compute the probability of an algorithm's solution lying in a ball centered at the unique global minimizer, defined by $$\mathbb{P}[\| x - x^* \| \leq r]$$, where $$r$$ is the radius of the ball. We set $$r= 0.1$$ in the experiment.

For the quadratic programming (QP) benchmark, we estimate the probability of the local solution lying in a sublevel set, defined by $$\mathbb{P}[f(x) - f(x^*) \leq 0.01]$$, where $$x^*$$ is a global solution returned by Gurobi. This metric is applicable when uniqueness of the global minima is not guaranteed.

A disadvantage of a success probability metric is that it requires us to know where the global optimum is. In the case of QP, problems embeddable on the DWave device are still of low enough dimension that branch-and-bound methods (such as Gurobi) can solve them to provable optimality. However, for real-world functions or higher dimension examples where no algorithm has shown exceedingly good performance, this metric cannot be calculated.

## Time to Solution (TTS)
We use a time-to-solution (TTS) metric to compare the performance of the optimization algorithms. TTS is the number of runs or shots required to obtain the correct global solution at least once with 0.99 success probability.

> $$TTS = t_f \times \Big\lceil\frac{\ln(1-0.99)}{\ln(1-p_s)}\Big\rceil$$

Here $$t_f$$ is the average runtime per trial or shot, and $$p_s$$ is the success probability of finding the global solution in a given trial/shot. We regard a given result $$x_f$$ as a global solution if $$\vert f(x_f) - f(x^*) \vert \le 0.01$$, where $$x^*$$ is the solution returned by Gurobi.

<!-- ## Average Objective Value
Since the average objective value over many randomized trials can always be calculated, this metric is useful when the global minimum is not known or when a low but possibly not globally optimal function value is be acceptable.

In the QP problems, for classical methods, this metric is derived from paths taken from (the same) 1000 starting points given to each algorithm. The values from the quantum methods run on the DWave [DWAVE MACHINE NAME] are obtained from 1000 shots (repetitions of the DWave experiment), followed by gradient descent to the nearest local minimum in the domain, and is done by an augmented Lagrangian method.

For the 2D benchmark, the classical algorithms are also each started from the same set of 1000 randomly generated points. The classical algorithms are again averaged by their function value at termination. However, since the quantum algorithms are simulated in this case, the average function value is obtained directly from the state vector's corresponding probability distribution and evaluation of the objective function at each point on the domain. -->


<!-- ### Metrics for evaluation

Because quantum measurement is a random process, we develop several metrics for tracking the behavior of QHD for comparison with other methods. With the full state vector available at every step of the algorithm, we can compute the exact measurement probability distribution.

- **Expected Energy**:
  - Quantum: The expected value of energy is obtained by taking the product of the wave function's probability distribution with the function values over the discretized domain. This can be done at every step of the evolution.
  - Classical: The expected value of energy can be obtained from averaging the function value at each step over the paths. If a path converges early, its function values continues to count towards the average from the (possibly suboptimal) minima.

- **Neighborhood Probability**:
  - Quantum: Using the final probability distribution, we can take the sum of probabilities for all outcomes within a radius of the global minimum's (known) location.
  - Classical: The expected value of energy can be obtained from averaging the function value at each step over the paths. If a path converges early, its final position continues to count towards the neighborhood probability from the (possibly suboptimal) minima. -->
