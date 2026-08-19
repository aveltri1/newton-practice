def optimize(x0, f, epsilon = .001):
    def f_prime(f, x0, epsilon):
        deriv = (f(x0 + epsilon) - f(x0 - epsilon)) / epsilon
        return deriv

    def f_second(f, x0, epsilon):
        deriv2 = (f_prime(f, x0 + epsilon, epsilon) -
                  f_prime(f, x0 - epsilon, epsilon)) / epsilon
        return deriv2

    for i in range(100):
        x = x0 - f_prime(f, x0, epsilon) / f_second(f, x0, epsilon)

        if abs(x - x0) < epsilon:
            break

        x0 = x

    return x
