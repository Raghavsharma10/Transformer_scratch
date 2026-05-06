def sbesselj(x, N):
    """Returns a vector of spherical bessel functions jn:

        x:   The argument.
        N:   values of n will run from 0 to N-1.

    """

    nmax = N - 1;
    out = np.zeros(N, dtype=np.float64)
    z = x ** 2

    out[0] = np.sin(x) / x
    j1 = np.sin(x) / z - np.cos(x) / x

    u = 1
    v = x / (2.0 * nmax + 1.0)
    w = v
    n = nmax

    while(np.abs(v / w) > 1e-20):
        n = n + 1
        u = 1 / (1 - z * u / (4.0 * n ** 2 - 1.0))
        v *= u - 1
        w += v

    out[nmax] = w

    for n in xrange(nmax - 1, 0, -1):
        out[n] = 1.0 / ((2.0 * n + 1.0) / x - out[n + 1])

    if(np.abs(out[0]) >= np.abs(j1)):
        out[1] *= out[0]
    else:
        out[1] = j1

    for n in xrange(1, nmax):
        out[n + 1] *= out[n]

    return out