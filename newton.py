def optimize(x0, f, epsilon=0.001):
     """
    Find an approximate optimum of a function using Newton's method.

    Parameters
    ----------
    x0 : float
        Starting value for x.
    f : function
        Function to optimize.
    epsilon : float
        Small value used to approximate the derivatives and determine
        when the algorithm has converged.

    Returns
    -------
    float
        Approximate x-value of the optimum.
    """ 
    def f_prime(f, x0, epsilon):
        """
        Approximate the first derivative of a function using finite differences.
        """
        deriv = (f(x0 + epsilon) - f(x0 - epsilon)) / epsilon
        return deriv

    def f_second(f, x0, epsilon):
        """
        Approximate the first derivative of a function using finite differences.
        """
        deriv2 = (
            f_prime(f, x0 + epsilon, epsilon) - f_prime(f, x0 - epsilon, epsilon)
        ) / epsilon
        return deriv2

    for i in range(100):
        x = x0 - f_prime(f, x0, epsilon) / f_second(f, x0, epsilon)

        if abs(x - x0) < epsilon:
            break

        x0 = x

    return x
