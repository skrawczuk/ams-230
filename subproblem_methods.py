from trust_region import TrustRegion
import numpy as np


class DogLeg(TrustRegion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def solve_subproblem(self, x, delta):
        g = self.grad_cost(x)
        b = self.b(x)

        if g.T @ b @ g <= 0:
            p = -delta / np.linalg.norm(g) * g
        else:
            p_u = (g.T @ g).squeeze() / (g.T @ b @ g) * g
            p_b = - np.linalg.inv(b) @ g

            if p_u.T @ (p_b - p_u) <= 0:
                p = -delta / np.linalg.norm(g) * g

            else:
                tau = self.calculate_tau(p_u, p_b, delta)
                p = p_u + (tau - 1) * (p_b - p_u)
        return p

    @staticmethod
    def calculate_tau(p_u, p_b, delta):
        a = np.linalg.norm(p_b - p_u) ** 2
        b = 2 * ((p_b - p_u).T @ p_u).squeeze()
        c = np.linalg.norm(p_u) ** 2 - delta ** 2
        return 1 + (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
