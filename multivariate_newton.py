import numpy as np
from scipy.differentiate import hessian


def multivariate_newton(F, x0, eps, max_iter = 100):
    x = np.asarray(x0)
    for i in range(max_iter):
        grad = jacobian(F, x).df
        H = hessian(F, x).ddf
        iter1 = np.linalg.solve(H, grad)
        x_new = x + iter1

        if np.linalg.norm(iter1) < eps:
            return x_new, i + 1

        x = x_new

    return x, max_iter

    