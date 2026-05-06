def DampedPowerlaw(q, a, alpha, sigma):
    """Damped power-law

    Inputs:
    -------
        ``q``: independent variable
        ``a``: factor
        ``alpha``: exponent
        ``sigma``: hwhm of the damping Gaussian

    Formula:
    --------
        ``a*q^alpha*exp(-q^2/(2*sigma^2))``
    """
    return a * q ** alpha * np.exp(-q ** 2 / (2 * sigma ** 2))