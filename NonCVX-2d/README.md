# Quantum Hamiltonian Descent

Quantum Hamiltonian Descent (QHD) is a novel physically-inspired quantum computing framework for continuous optimization problems.



## Benchmark Set: introducing the class 'Experiment2D'

We build a benchmark set for non-convex numerical optimization. This set contains 22 objective functions (mostly non-convex), each with a **unique** global minimum (but possibly many local minima). Raw data of these functions can be found in "/utils/experimentSetup.m": for each function, we include its analytical formula, domain of definition/optimization, coordinates of the global minizer.



For the sake of comparable performance in the numerical simulation of Schrodinger equations (PDE), we *normalize* all objective functions so that

- they are now defined on the unit square $[0,1]^2$;
- their Lipschitz constants and Hessian condition number does not change;
- their global minimum is 0;
-  Dirichlet boundary condition is applied.

To do so, for a function $f(x_1,x_2)$ originally defined on a square domain $[a,b]^2$ with global minimizer $x^\ast = (x^\ast_1, x^\ast_2)$, we transform it to be $F(y_1,y_2) := \frac{1}{L} f(a+Ly_1,a+Ly_2) - f_{\min}$, with $L=b-a$. The global minimizer for $F(y_1,y_2)$ is $y^\ast_1 = (x^\ast_1 - a)/L$, $y^\ast_2 = (x^\ast_2 - b)/L$. By the chain rule, the gradient of $F(y_1,y_2)$ is 
$$\nabla F(y) = \nabla f(a + L y).$$


This normalization has been implemented in "pdeexperiments/utils/@Experiment2D.m", which defines a new class **Experiment2D** for the purpose of automated benchmark testing. **Experiment2D** is a MATLAB class that specifies a 2-dimensional experiment instance with the following properties and methods:

1. Properties:

- *experiment_dir*: the short name for the function;
- *experiment_V_str*: the full name for the function;
- *bounds*: a 1x2 cell array {a, b}$;
- *L*: the edge length of the original square domain, L = b-a;
- *cells_along_dim*: number of cells in the discretization grid along each dimension;
- *step_size*: size of each grid cell, step_size = 1/cells_along_dim (**normalized**);
- *X1, X2*: meshgrid on both coordinates (**normalized**);
- *potential_fn*: a function handle of f(x1,x2);
- *global_min*: a 1x2 double array <img src="https://render.githubusercontent.com/render/math?math=[x^*, y^*]"> (**normalized**);
- *nbhd_radius*: a positive double scalar indicating the effective neighborhood of the global minimizer (**normalized**);
- *lipschitz*: the (approximated) lipschitz constant of the function (**invariant of dilation**);
- *V*: function value over the meshgrid [X1, X2] (**normalized**);
- *H_T*: the kinetic Hamiltonian (**normalized**);
- *H_U*: the **diagonal** of the potential Hamiltonian (**normalized**). <img src="https://render.githubusercontent.com/render/math?math=\langle\psi, H_U .* \psi \rangle"> returns the **potential energy/loss function**.
- *mesh_ind*: the index mesh of the "neighborhood" of the global minimizer (**invariant of dilation**).
- *H_ind*: the **diagonal** of the index matrix of the "neighborhood" of the global minimizer. <img src="https://render.githubusercontent.com/render/math?math=\langle\psi, H_{ind} .* \psi \rangle"> returns the **success probability** (**invariant of dilation**).

2. Methods:

- *Experiment2D*: [Constructor methods] initialization of a class object;

- *feval*: function evaluation oracle of the **normalized** function;

- *grad*: gradient oracle of the **normalized** function.

- *get_plot*: plot the surface of the objective function and a red circle indicating the neighborhood of the global minimizer.

  

  For a time-dependent Schrodinger equation with the QHD Hamiltonian <img src="https://render.githubusercontent.com/render/math?math=H(t) = -\frac{1}{2}\varphi(t)\nabla^2 + V(x)"> defined on the unit square with vanishing Dirichlet boundary condition, we provide two solvers (which are included in methods of **Experiment2D**)

- *fdmleapfrog*: a centered finite difference scheme + Leapfrog time propagation;

- *pseudospec*: a pseudospectral spatial discretization scheme + 2nd order Trotter.
