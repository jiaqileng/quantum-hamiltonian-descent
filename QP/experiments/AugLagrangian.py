import numpy as np

class AugLagrangian():
    """
    This class implements the augmented Lagrangian method for the quadratic programming problem:
            
            min x^T Q x + b^T x, subject to Q_c x = b_c and lb <= x <= ub
            
    n = dimension of the problem, m = number of equality constraints.
    Q = 2d array, size = (n,n)
    b = 1d array, size = (1,n)
    Q_c = 2d array, size = (m,n)
    b_c = 1d array, size = (1,m)
    lb, ub = scalar values
    """
    def __init__(self, Q, b, Q_c, b_c, lb, ub):
        super().__init__()
        
        self.Q = Q
        self.b = b
        self.Q_c = Q_c
        self.b_c = b_c
        self.lb = lb
        self.ub = ub 
        self.dimension = len(b)
        self.n_constraints = len(b_c)
    
    def augmented_lagrangian(self, x, mu, lbd):
        objective = 0.5 * x @ self.Q @ x + self.b @ x
        penalty = 0.5 * mu * np.sum((self.Q_c @ x - self.b_c)**2)
        lag_multiplier = np.sum(lbd * (self.Q_c @ x - self.b_c))

        return objective + penalty + lag_multiplier
    
    def get_grad(self, x, mu, lbd):
        obj_grad = self.Q @ x + self.b
        penalty_grad = mu * (self.Q_c.transpose() @ self.Q_c @ x - self.b_c @ self.Q_c)
        lag_mult_grad = lbd @ self.Q_c
        
        return obj_grad + penalty_grad + lag_mult_grad

    def optimizer(self, x0, max_iter, penalty_base, tol, eta, lr0=1e-1):
        """
        Args:
            x0: initial guess
            max_iter: maximal number of iterations per round
            penalty_base: base number used in the penalty coeff
            tol: tolerance of gradient norm
            eta: tolerance of the violation of equality constraints
        Returns:
            result: final result (approx. minimizer of the constrained opt problem)
        """
        x = x0
        lbd = np.zeros(self.n_constraints)
        loss_curve = []
        if self.Q_c.size > 0:
            
            round_number = 1

            while np.linalg.norm(self.Q_c @ x - self.b_c) > eta:
                # penalty coeff
                mu = penalty_base**(round_number)
                lr = lr0 / mu
            
                # Reset values needed for each optimization run
                step = 0
                gradient = float("inf") * np.ones_like(x)
                
                # Run GD
                while np.linalg.norm(gradient) > tol and step < max_iter:   
                    # Lagrangian multiplier
                    
                    lbd += mu * (self.Q_c @ x - self.b_c)
                    # GD update
                    gradient = self.get_grad(x, mu, lbd)
                    x -= lr * gradient
                    x = np.clip(x, self.lb, self.ub)
                    fval = self.augmented_lagrangian(x, mu, lbd)
                    loss_curve.append(fval)
                    step += 1

                round_number += 1
            
            res = {"final_soln": x,
                "loss_curve": loss_curve}
            
            return res
        else:

            step = 0
            mu = 0
            lr = 5e-2
            gradient = float("inf") * np.ones_like(x)

            # Run GD
            while np.linalg.norm(gradient) > tol and step < max_iter:
                gradient = self.get_grad(x, mu, lbd)
                x -= lr * gradient
                x = np.clip(x, self.lb, self.ub)
                fval = self.augmented_lagrangian(x, mu, lbd)
                loss_curve.append(fval)
                step += 1
            
            res = {"final_soln": x,
                "loss_curve": loss_curve}
            
            return res