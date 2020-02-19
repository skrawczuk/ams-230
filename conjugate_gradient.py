from line_search import LineSearch
import numpy as np


class ConjugateGradient:
    def __init__(self, A, b):
        self.A = A
        self.b = b

    def minimize(self, x0, tol=1e-8):
        a, p, k = self.step(x0, eps=tol)
        x = x0 + a * p

        return x, k

    def step(self, x, eps):
        x = x[:, None]
        x = x.reshape(-1,1)

        r_new = self.A @ x - self.b
        p = -r_new
        k = 0
        alpha = 0
        while np.linalg.norm(r_new, 2) > eps:
            r = r_new

            alpha = (r.T @ r) / (p.T @ self.A @ p)
            x += alpha * p
            r_new = r + alpha * self.A @ p

            beta = (r_new.T @ r_new) / (r.T @ r)
            p = -r_new + beta * p
            k += 1

        return alpha, p.T, k


class FletcherReeves(LineSearch):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def minimize(self, x0, tol=1e-8, verbose=False):
        x = np.zeros((self.max_iterations + 1, x0.shape[0]))
        x[0] = x0
        df = self.grad_cost(x0)
        p = -df

        i = 0
        while np.linalg.norm(df) > tol and i < self.max_iterations:
            i += 1
            a = self.step_length(p, x[i - 1])
            x[i] = x[i - 1] + (a * p).reshape(-1)

            df_new = self.grad_cost(x[i])
            b = (df_new.T @ df_new) / (df.T @ df)
            p = -df_new + b * p

            df = df_new

            if verbose:
                if i % verbose == 0:
                    print('iter', i, 'of', self.max_iterations)

        return x[:i + 1]
