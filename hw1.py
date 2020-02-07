from line_search import LineSearch
import matplotlib.pyplot as plt
import numpy as np


def f(x, c):
    x1, x2 = x.squeeze()
    return (c*x1 - 2)**4 + x2**2 * (c*x1 - 2)**2 + (x2 + 1)**2


def grad_f(x, c):
    x1, x2 = x.squeeze()
    return np.array([[
        4*c * (c*x1 - 2)**3 + 2*c*x2**2 * (c*x1 - 2),
        2*x2 * (c*x1 - 2)**2 + 2*x2 + 2
    ]])


if __name__ == '__main__':
    max_iterations = 2000
    x0 = np.random.random(2)
    tol = 1e-8
    c1 = 0.1
    c2 = 0.9
    a_max = 1e6

    xc1 = LineSearch(f, grad_f, max_iterations, c=1).minimize(
        x0, tol, c1, c2, a_max)

    xc10 = LineSearch(f, grad_f, max_iterations, c=10).minimize(
        x0, tol, c1, c2, a_max)

    plt.plot(np.linalg.norm(xc1 - np.array([2, -1]), axis=1), label='c=1');
    plt.plot(np.linalg.norm(xc10 - np.array([.2, -1]), axis=1), label='c=10');

    plt.yscale('log')
    plt.ylabel('$log |x - x*|$')
    plt.xlabel('iterations')
    plt.legend()
    plt.grid()
    plt.show()
