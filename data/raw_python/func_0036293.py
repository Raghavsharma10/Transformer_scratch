def sbesselj_sum(z, N):
    """Tests the Spherical Bessel function jn using the sum:

        Inf
        sum  (2*n+1) * jn(z)**2 = 1
        n=0


        z:  The argument.
        N:  Large N value that the sum runs too.

    Note that the sum only converges to 1 for large N value (i.e. N >> z).

    The routine returns the relative error of the assumption.
    """

    b = sbesselj(z, N)
    vvv = 2.0 * np.array(range(0, N), dtype=np.float64) + 1.0
    sm = np.sum(np.sort(vvv * (b ** 2)))
    return np.abs((sm - 1.0) / sm) + np.spacing(1)