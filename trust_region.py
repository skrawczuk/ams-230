import numpy as np


class TrustRegion:
    def __init__(self, cost_function, grad_function, max_iterations, **kwargs):
        self.cost = lambda x: cost_function(x, **kwargs)
        self.grad_cost = lambda x: grad_function(x, **kwargs)
        self.max_iterations = max_iterations

        hess_f = kwargs.pop('hess_f')
        self.b = lambda x: hess_f(x, **kwargs)
        self.m = lambda p, x: self.cost(x) + self.grad_cost(x).T @ p + p.T @ self.b(x) @ p

    def minimize(self, x0, delta_max, eta=1/8, verbose=0):
        delta = delta_max / 2
        x = np.zeros((self.max_iterations + 1, x0.shape[0]))
        x[0] = x0

        for i in range(self.max_iterations):
            p = self.solve_subproblem(x[i], delta)
            rho = (self.cost(x[i]) - self.cost(x[i] + p.squeeze())) / (self.m(np.zeros_like(p), x[i]) - self.m(p, x[i]))
            if np.isnan(rho):
                break

            if rho < 1 / 4:
                delta /= 4
            else:
                if rho > 3 / 4 and np.linalg.norm(rho) == delta:
                    delta = min(2 * delta, delta_max)

            if rho > eta:
                x[i+1] = x[i] + p.squeeze()
            else:
                x[i+1] = x[i]

            if verbose:
                if i % verbose == 0:
                    print('iter', i, 'of', self.max_iterations)

        return x[:i+1]

    def solve_subproblem(self, x, delta):
        """
        Cauchy point method for subproblem
        """
        g = self.grad_cost(x)
        b = self.b(x)
        tau = self.calculate_tau(g, b, delta)

        return -tau * delta / np.linalg.norm(g) * g

    @staticmethod
    def calculate_tau(g, b, delta):
        if g.T @ b @ g <= 0:
            return 1.
        else:
            return min(np.linalg.norm(g) ** 3 / (delta * g.T @ b @ g), 1.)
