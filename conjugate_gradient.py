from line_search import LineSearch
import numpy as np


class ConjugateGradient:
    """
    conjugate gradient using exact line search
    """
    def __init__(self, A, b):
        self.A = A
        self.b = b

    def minimize(self, x0, tol=1e-8):
        a, p, k = self.step(x0, eps=tol)
        x = x0 + a * p

        return x, k

    def step(self, x, eps):
        x = x[:, None]

        r_new = self.A @ x - self.b
        p = -r_new
        k = 0
        alpha = 0
        while np.linalg.norm(r_new) / r_new.size > eps:
            r = r_new

            alpha = (r.T @ r) / (p.T @ self.A @ p)
            x += alpha * p
            r_new = r + alpha * self.A @ p

            beta = (r_new.T @ r_new) / (r.T @ r)
            p = -r_new + beta * p
            k += 1

        return alpha, p.T, k


class FletcherReeves(LineSearch):
    """
    F-R Conjugate gradient method with inexact line search
    """
    def __init__(self, *args, **kwargs):
        self.restart = kwargs.pop('restart')
        super().__init__(*args, **kwargs)

    def minimize(self, x0, tol=1e-8, verbose=False):
        x = np.zeros((self.max_iterations + 1, x0.shape[0]))
        x[0] = x0
        df = self.grad_cost(x0)
        p = -df

        i = 0
        while np.linalg.norm(df)/df.size > tol and i < self.max_iterations:
            i += 1
            a = self.step_length(p, x[i - 1])
            x[i] = x[i - 1] + (a * p).reshape(-1)

            df_new = self.grad_cost(x[i])
            if self.restart and np.abs(df_new @ df) / np.linalg.norm(df_new)**2 >= 0.1:
                b = 0
            else:
                b = (df_new.T @ df_new) / (df.T @ df)
            p = -df_new + b * p

            df = df_new

            if verbose:
                if i % verbose == 0:
                    print('iter', i, 'of', self.max_iterations, ': ', np.linalg.norm(df)/df.size)

        return x[:i + 1]
