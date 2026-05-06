def yn2Kn2Der(nu, y, n=0, tol=5e-4, nterms=1, nu_step=0.001):
    r"""Computes the function :math:`y^{\nu/2} K_{\nu}(y^{1/2})` and its derivatives.
    
    Care has been taken to handle the conditions at :math:`y=0`.
    
    For `n=0`, uses a direct evaluation of the expression, replacing points
    where `y=0` with the appropriate value. For `n>0`, uses a general sum
    expression to evaluate the expression, and handles the value at `y=0` using
    a power series expansion. Where it becomes infinite, the infinities will
    have the appropriate sign for a limit approaching zero from the right.
    
    Uses a power series expansion around :math:`y=0` to avoid numerical issues.
    
    Handles integer `nu` by performing a linear interpolation between values of
    `nu` slightly above and below the requested value.
    
    Parameters
    ----------
    nu : float
        The order of the modified Bessel function and the exponent of `y`.
    y : array of float
        The points to evaluate the function at. These are assumed to be
        nonegative.
    n : nonnegative int, optional
        The order of derivative to take. Set to zero (the default) to get the
        value.
    tol : float, optional
        The distance from zero for which the power series is used. Default is
        5e-4.
    nterms : int, optional
        The number of terms to include in the power series. Default is 1.
    nu_step : float, optional
        The amount to vary `nu` by when handling integer values of `nu`. Default
        is 0.001.
    """
    n = int(n)
    y = scipy.asarray(y, dtype=float)
    
    if n == 0:
        K = y**(nu / 2.0) * scipy.special.kv(nu, scipy.sqrt(y))
        K[y == 0.0] = scipy.special.gamma(nu) / 2.0**(1.0 - nu)
    else:
        K = scipy.zeros_like(y)
        for k in scipy.arange(0.0, n + 1.0, dtype=float):
            K += (
                scipy.special.binom(n, k) * fixed_poch(1.0 + nu / 2.0 - k, k) *
                y**(nu / 2.0 - k) * Kn2Der(nu, y, n=n-k)
            )
        # Do the extra work to handle y == 0 only if we need to:
        mask = (y == 0.0)
        if (mask).any():
            if int(nu) == nu:
                K[mask] = 0.5 * (
                    yn2Kn2Der(nu - nu_step, y[mask], n=n, tol=tol, nterms=nterms, nu_step=nu_step) +
                    yn2Kn2Der(nu + nu_step, y[mask], n=n, tol=tol, nterms=nterms, nu_step=nu_step)
                )
            else:
                if n > nu:
                    K[mask] = scipy.special.gamma(-nu) * fixed_poch(1 + nu - n, n) * scipy.inf
                else:
                    K[mask] = scipy.special.gamma(nu) * scipy.special.gamma(n + 1.0) / (
                        2.0**(1.0 - nu + 2.0 * n) * fixed_poch(1.0 - nu, n) *
                        scipy.special.factorial(n)
                    )
    if tol > 0.0:
        # Replace points within tol (absolute distance) of zero with the power
        # series approximation:
        mask = (y <= tol) & (y > 0.0)
        K[mask] = 0.0
        if int(nu) == nu:
            K[mask] = 0.5 * (
                yn2Kn2Der(nu - nu_step, y[mask], n=n, tol=tol, nterms=nterms, nu_step=nu_step) +
                yn2Kn2Der(nu + nu_step, y[mask], n=n, tol=tol, nterms=nterms, nu_step=nu_step)
            )
        else:
            for k in scipy.arange(n, n + nterms, dtype=float):
                K[mask] += (
                    scipy.special.gamma(nu) * fixed_poch(1.0 + k - n, n) * y[mask]**(k - n) / (
                        2.0**(1.0 - nu + 2 * k) * fixed_poch(1.0 - nu, k) * scipy.special.factorial(k))
                    )
            for k in scipy.arange(0, nterms, dtype=float):
                K[mask] += (
                    scipy.special.gamma(-nu) * fixed_poch(1.0 + nu + k - n, n) *
                    y[mask]**(nu + k - n) / (
                        2.0**(1.0 + nu + 2.0 * k) * fixed_poch(1.0 + nu, k) *
                        scipy.special.factorial(k)
                    )
                )
    
    return K