from line_search import LineSearch
import numpy as np


class FletcherReeves(LineSearch):
    """
    F-R Conjugate gradient method with inexact line search
    """
    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        restart : bool
            allow reset of beta during minimization
        PR : bool
            use Polak–Ribiere for calculating beta. Standard method if False
        """
        self.restart = kwargs.pop('restart')
        self.beta_function = self.beta_pr if 'PR' in kwargs and kwargs.pop('PR') else self.beta

        super().__init__(*args, **kwargs)

    def minimize(self, x0, tol=1e-8, verbose=False):
        x = np.zeros((self.max_iterations + 1, x0.shape[0]))
        x[0] = x0
        df = self.grad_cost(x0)
        p = -df

        i = 0
        while np.linalg.norm(df) / df.size > tol and i < self.max_iterations:
            i += 1
            a = self.step_length(p, x[i - 1])
            x[i] = x[i - 1] + (a * p).reshape(-1)

            df_new = self.grad_cost(x[i])
            if self.restart and np.abs(df_new @ df) / np.linalg.norm(df_new)**2 >= 0.1:
                b = 0
            else:
                b = self.beta_function(df, df_new)
            p = -df_new + b * p

            df = df_new

            if verbose:
                if i % verbose == 0:
                    print('iter', i, 'of', self.max_iterations, ': ', np.linalg.norm(df) / df.size)

        return x[:i + 1]

    @staticmethod
    def beta(df, df_new):
        return (df_new.T @ df_new) / (df.T @ df)

    @staticmethod
    def beta_pr(df, df_new):
        return df_new.T @ (df_new - df) / np.linalg.norm(df)**2


def conjugate_gradient(A, b, x0, max_iterations, tol=1e-8):
    """
    Conjugate gradient using exact line search on f(x) = 1/2 x^T A x - b^T x

    Parameters
    ----------
    A : array_like
        A matrix in quadratic function
    b : array_like
        b matrix in quadratic function
    x0 : array_like
        initial guess
    max_iterations : int
        maximum iterations
    tol : float
        stopping tolerance for gradient magnitude
    """
    x = np.zeros((max_iterations + 1, x0.shape[0]))
    x[0] = x0
    r_new = A @ x[0][:, None] - b
    p = -r_new
    i = 0
    while np.linalg.norm(r_new) / r_new.size > tol and i < max_iterations:
        i += 1
        r = r_new

        a = (r.T @ r) / (p.T @ A @ p)
        x[i] = x[i-1] + a * p.T
        r_new = r + a * A @ p

        beta = (r_new.T @ r_new) / (r.T @ r)
        p = -r_new + beta * p

    x[i+1] = x[i] + a * p.T
    return x[:i+2]
