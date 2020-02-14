from line_search import LineSearch
import numpy as np


class Newton(LineSearch):
    def __init__(self, *args, **kwargs):
        hess_cost = kwargs.pop('hess_f')
        self.hess_cost = lambda x: hess_cost(x, **kwargs)

        super().__init__(*args, **kwargs)

    def step_direction(self, x):
        return (-np.linalg.inv(self.hess_cost(x)) @ self.grad_cost(x).T).T


class ConjugateGradient(LineSearch):
    def __init__(self, *args, **kwargs):
        self.A = kwargs['A']
        self.b = kwargs['b']

        super().__init__(*args, **kwargs)

    def step_direction(self, x, eps=1e-8):
        x = x[:, None]

        r_new = self.A @ x - self.b
        p = -r_new

        while np.all(r_new > eps):
            r = r_new
            rr = r.T @ r
            ap = self.A @ p

            alpha = rr / (p.T @ ap)
            x += alpha * p
            r_new = r - alpha * ap

            beta = (r_new.T @ r_new) / rr
            p = -r + beta * p

            if np.all(r < eps):
                break

        return p.T
