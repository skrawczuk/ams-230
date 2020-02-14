from line_search import LineSearch
import numpy as np


class Newton(LineSearch):
    def __init__(self, *args, **kwargs):
        hess_cost = kwargs.pop('hess_f')
        self.hess_cost = lambda x: hess_cost(x, **kwargs)

        super().__init__(*args, **kwargs)

    def step_direction(self, x):
        return (-np.linalg.inv(self.hess_cost(x)) @ self.grad_cost(x).T).T