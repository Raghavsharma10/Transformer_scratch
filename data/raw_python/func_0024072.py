def Kn2Der(nu, y, n=0):
    r"""Find the derivatives of :math:`K_\nu(y^{1/2})`.
    
    Parameters
    ----------
    nu : float
        The order of the modified Bessel function of the second kind.
    y : array of float
        The values to evaluate at.
    n : nonnegative int, optional
        The order of derivative to take.
    """
    n = int(n)
    y = scipy.asarray(y, dtype=float)
    sqrty = scipy.sqrt(y)
    if n == 0:
        K = scipy.special.kv(nu, sqrty)
    else:
        K = scipy.zeros_like(y)
        x = scipy.asarray(
            [
                fixed_poch(1.5 - j, j) * y**(0.5 - j)
                for j in scipy.arange(1.0, n + 1.0, dtype=float)
            ]
        ).T
        for k in scipy.arange(1.0, n + 1.0, dtype=float):
            K += (
                scipy.special.kvp(nu, sqrty, n=int(k)) *
                incomplete_bell_poly(n, int(k), x)
            )
    return K