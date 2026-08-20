import numpy as np
from scipy.differentiate import hessian


def multivariate_newton(F, x0, eps, max_iter = 100):
    x = np.asarray(x0)
    for i in range(max_iter):
        grad = jacobian(F, x)
        H = hessian(F, x).ddf
        iter1 = np.linalg.solve(H, grad)
        xnew = x + iter1

        if np.linalg.norm(iter1) < eps:
            return xnew, i + 1

        x = xnew

    return x, max_iter

    