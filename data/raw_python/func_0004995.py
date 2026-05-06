def BorueErukhimovich_Powerlaw(q, C, r0, s, t, nu):
    """Borue-Erukhimovich model ending in a power-law.

    Inputs:
    -------
        ``q``: independent variable
        ``C``: scaling factor
        ``r0``: typical el.stat. screening length
        ``s``: dimensionless charge concentration
        ``t``: dimensionless temperature
        ``nu``: excluded volume parameter

    Formula:
    --------
        ``C*(x^2+s)/((x^2+s)(x^2+t)+1)`` where ``x=q*r0`` if ``q<qsep``
        ``A*q^(-1/nu)``if ``q>qsep``
        ``A`` and ``qsep`` are determined from conditions of smoothness at the
        cross-over.
    """
    def get_xsep(alpha, s, t):
        A = alpha + 2
        B = 2 * s * alpha + t * alpha + 4 * s
        C = s * t * alpha + alpha + alpha * s ** 2 + alpha * s * t - 2 + 2 * s ** 2
        D = alpha * s ** 2 * t + alpha * s
        r = np.roots([A, B, C, D])
        #print "get_xsep: ", alpha, s, t, r
        return r[r > 0][0] ** 0.5
    get_B = lambda C, xsep, s, t, nu:C * (xsep ** 2 + s) / ((xsep ** 2 + s) * (xsep ** 2 + t) + 1) * xsep ** (1.0 / nu)
    x = q * r0
    xsep = np.real_if_close(get_xsep(-1.0 / nu, s, t))
    A = get_B(C, xsep, s, t, nu)
    return np.piecewise(q, (x < xsep, x >= xsep),
                        (lambda a:BorueErukhimovich(a, C, r0, s, t),
                         lambda a:A * (a * r0) ** (-1.0 / nu)))