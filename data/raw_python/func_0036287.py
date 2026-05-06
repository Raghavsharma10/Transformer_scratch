def sbessely(x, N):
    """Returns a vector of spherical bessel functions yn:

        x:   The argument.
        N:   values of n will run from 0 to N-1.

    """

    out = np.zeros(N, dtype=np.float64)

    out[0] = -np.cos(x) / x
    out[1] = -np.cos(x) / (x ** 2) - np.sin(x) / x

    for n in xrange(2, N):
        out[n] = ((2.0 * n - 1.0) / x) * out[n - 1] - out[n - 2]

    return out