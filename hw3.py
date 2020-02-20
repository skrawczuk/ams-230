from conjugate_gradient import conjugate_gradient, FletcherReeves
import numpy as np
import matplotlib.pyplot as plt


def problem3():
    n = int(1e3)

    eigs1 = np.logspace(1, 3, n)
    eigs2 = np.append(2 * np.random.random(int(n / 2)) + 9, 2 * np.random.random(int(n / 2)) + 999)

    Q, R = np.linalg.qr(np.random.random((n, n)))
    D1 = np.diag(eigs1)
    D2 = np.diag(eigs2)

    A1 = Q.T @ D1 @ Q
    A2 = Q.T @ D2 @ Q
    b = np.random.random((n, 1))
    x0 = np.random.random(n)
    max_iterations = 2000
    tol = 1e-8

    x1 = conjugate_gradient(A1, b, x0, max_iterations, tol)
    x2 = conjugate_gradient(A2, b, x0, max_iterations, tol)

    plt.plot(np.linalg.norm(x1[1:] - x1[:-1], axis=1), label='uniform eigenvalues')
    plt.plot(np.linalg.norm(x2[1:] - x2[:-1], axis=1), label='clustered eigenvalues')
    plt.yscale('log')
    plt.ylabel('||$\Delta x$||')
    plt.xlabel('iterations')
    plt.legend()
    plt.title('Problem 3')
    plt.grid()
    plt.show()


def problem4():
    import autograd.numpy as np
    from autograd import grad

    def f(x):
        s = 0
        for i in range(len(x) - 1):
            s += 100 * (x[i] ** 2 - x[i + 1]) ** 2 + (x[i] - 1) ** 2
        return s

    def grad_f(x):
        return grad(f)(x)

    x0 = np.random.random(100)
    tol = 1e-6
    x1 = FletcherReeves(f, grad_f, max_iterations=1100, restart=False, PR=False).minimize(x0, tol)
    x2 = FletcherReeves(f, grad_f, max_iterations=1100, restart=True, PR=False).minimize(x0, tol)
    x3 = FletcherReeves(f, grad_f, max_iterations=1100, restart=False, PR=True).minimize(x0, tol)

    c1 = np.array([f(x1[i]) for i in range(len(x1))])
    c2 = np.array([f(x2[i]) for i in range(len(x2))])
    c3 = np.array([f(x3[i]) for i in range(len(x3))])

    plt.plot(np.linalg.norm(c1[1:] - c1[:-1], axis=1), label='FR')
    plt.plot(np.linalg.norm(c2[1:] - c2[:-1], axis=1), label='FR w/ restart')
    plt.plot(np.linalg.norm(c3[1:] - c3[:-1], axis=1), label='PR')
    plt.ylabel('||$\Delta x$||')
    plt.xlabel('iterations')
    plt.legend()

    plt.show()


if __name__ == '__main__':
    problem4()
