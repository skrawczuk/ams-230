from line_search import LineSearch
from direction_methods import Newton
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


def problem2():
    def f(x):
        x1, x2 = x.squeeze()
        return 100 * (x2 - x1 ** 2) ** 2 + (1 - x1) ** 2

    def grad_f(x):
        x1, x2 = x.squeeze()
        return np.array([[
            2 * (200 * x1 ** 3 - 200 * x1 * x2 + x1 - 1),
            200 * (x2 - x1 ** 2)
        ]])

    def hess_f(x):
        x1, x2 = x.squeeze()
        return np.array([
            [-400 * (x2 - x1 ** 2) + 800 * x1 ** 2 + 2, -400 * x1],
            [-400 * x1, 200]
        ])

    max_iterations = 30000
    x01 = np.array([1.1, 1.2])
    x02 = np.array([1.2, 1.2])
    tol = 1e-8
    c1 = 0.1
    c2 = 0.9
    a_max = 1e6

    sd1 = LineSearch(f, grad_f, max_iterations).minimize(
        x01, tol, c1, c2, a_max, verbose=True)
    sd2 = LineSearch(f, grad_f, max_iterations).minimize(
        x02, tol, c1, c2, a_max, verbose=True)

    xn1 = Newton(f, grad_f, max_iterations, hess_f=hess_f).minimize(
        x01, tol, c1, c2, a_max)

    xn2 = Newton(f, grad_f, max_iterations, hess_f=hess_f).minimize(
        x02, tol, c1, c2, a_max)

    plt.plot(np.linalg.norm(xn1 - np.ones(2), axis=1), label='Newton $x0 = [1.1,1.2]$, {} iters'.format(len(xn1)))
    plt.plot(np.linalg.norm(xn2 - np.ones(2), axis=1), label='Newton $x0 = [1.2,1.2]$, {} iters'.format(len(xn2)))

    plt.plot(np.linalg.norm(sd1 - np.ones(2), axis=1), label='SD $x0 = [1.1,1.2]$, {} iters'.format(len(sd1)))
    plt.plot(np.linalg.norm(sd2 - np.ones(2), axis=1), label='SD $x0 = [1.2,1.2]$, {} iters'.format(len(sd2)))

    plt.title('Problem 2')
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel('$log |x - x*|$')
    plt.xlabel('iterations')
    plt.legend()
    plt.grid()
    plt.show()


def problem3():
    def f(x):
        x1, x2, x3 = x.squeeze()
        return 1/4 * x1**4 + 1/2 * (x2 - x3)**2 + 1/2*x2**2

    def grad_f(x):
        x1, x2, x3 = x.squeeze()
        return np.array([[
            x1**3, 2*x2 - x3, -x3
        ]])

    def hess_f(x):
        x1, x2, x3 = x.squeeze()
        return np.array([
            [3*x1**2, 0, 0],
            [0, 2, 0],
            [0, 0, -1]
        ])

    max_iterations = 10000
    x0 = 10*np.ones(3)
    tol = 1e-6
    c1 = 0.1
    c2 = 0.9
    a_max = 1e6

    x = Newton(f, grad_f, max_iterations, hess_f=hess_f).minimize(
        x0, tol, c1, c2, a_max, verbose=True)

    plt.title('Problem 3')
    plt.plot(np.abs(x))
    plt.yscale('log')
    plt.ylabel('$log |x - x*|$')
    plt.xlabel('iterations')
    plt.grid()
    plt.show()


def problem4():
    data = loadmat('/Users/schuylerkrawczuk/desktop/school/classes/ams-230/DATA.mat')

    x = data['DATA']
    L = data['LABELS']

    def p(x, L, theta):
        return 1 / (1 + np.exp(-L @ theta[None, :] @ x))

    def dp(x, L, theta):
        print(L.shape, theta[None, :].shape, x.shape)

        return (x[None, :].T @ L[:, None] * np.exp(-L @ theta[None, :] @ x) /
                (1 + np.exp(-L @ theta[None, :] @ x)) ** 2)

    def ddp(x, L, theta):
        return (2 * (x[:, None] @ L[:, None]) @ (x[:, None] @ L[:, None]).T *
                np.exp(-2 * L @ theta[None, :] @ x) / (1 + np.exp(-L @ theta[None, :] @ x)) ** 3)

    def f(theta):
        return (0.5 * np.linalg.norm(theta) ** 2 -
                np.sum([np.log(p(xi, Li, theta)) for xi, Li in zip(x, L)])).T

    def df(theta):
        return (theta -
                np.array([dp(xi, Li, theta) / p(xi, Li, theta) for xi, Li in zip(x, L)]).sum(axis=0).T)

    def ddf(theta):
        return 1 - np.array([(p(xi, Li, theta) * ddp(xi, Li, theta) - dp(xi, Li, theta).T @ dp(xi, Li, theta)) /
                             (p(xi, Li, theta) * p(xi, Li, theta))
                             for xi, Li in zip(x, L)]).sum(axis=0)

    theta_0 = 10*np.random.random(2)

    theta = LineSearch(f, df, 10).minimize(theta_0, 1e-7, verbose=0)
    theta_n = Newton(f, df, 10, hess_f=ddf).minimize(theta_0, 1e-7, verbose=0)

    plt.title('Problem 4')
    plt.xlabel('iterations')
    plt.ylabel('$\Theta$')
    plt.plot(theta[:10, 0], 'C0', label='SD')
    plt.plot(theta_n[:10, 0], 'C1', label='Newton')
    plt.plot(theta[:10, 1], 'C0')
    plt.plot(theta_n[:10, 1], 'C1')
    plt.legend()
    plt.grid()
    plt.show()



if __name__ == '__main__':

    problem4()