def Exponential(x, a, tau, y0):
    """Exponential function

    Inputs:
    -------
        ``x``: independent variable
        ``a``: scaling factor
        ``tau``: time constant
        ``y0``: additive constant

    Formula:
    --------
        ``a*exp(x/tau)+y0``
    """
    return np.exp(x / tau) * a + y0