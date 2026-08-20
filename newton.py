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
    tuple
        Approximate x-value, success/failure flag, and informative message.
    """

    # Check that f is a function
    if not callable(f):
        raise TypeError(
            f"Argument f must be a function, but is {type(f)}"
        )

    # Check that x0 is a number
    if not isinstance(x0, (int, float)):
        raise TypeError(
            f"Starting value x0 must be a number, but is {type(x0)}"
        )

    # Check that epsilon is a positive number
    if not isinstance(epsilon, (int, float)):
        raise TypeError(
            f"epsilon must be a number, but is {type(epsilon)}"
        )

    if epsilon <= 0:
        raise ValueError("epsilon must be greater than 0.")

    def f_prime(f, x0, epsilon):
        """
        Approximate the first derivative of a function using finite differences.
        """
        deriv = (f(x0 + epsilon) - f(x0 - epsilon)) / (2 * epsilon)
        return deriv

    def f_second(f, x0, epsilon):
        """
        Approximate the second derivative of a function using finite differences.
        """
        deriv2 = (
            f_prime(f, x0 + epsilon, epsilon)
            - f_prime(f, x0 - epsilon, epsilon)
        ) / (2 * epsilon)
        return deriv2

    for i in range(100):

        second_derivative = f_second(f, x0, epsilon)

        # Newton's method cannot continue if the second derivative is zero
        if second_derivative == 0:
            return (
                x0,
                False,
                "Newton's method failed because the second derivative is zero."
            )

        x = x0 - f_prime(f, x0, epsilon) / second_derivative

        # Warn if Newton's method takes a very large step
        if abs(x - x0) > 10:
            print(
                "Warning: Newton's method is taking a large step."
            )

        # Check convergence
        if abs(x - x0) < epsilon:
            return (
                x,
                True,
                "Newton's method successfully converged."
            )

        x0 = x

    # If we get here, the method did not converge within 100 iterations
    return (
        x0,
        False,
        "Newton's method did not converge within 100 iterations."
    )