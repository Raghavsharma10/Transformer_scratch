def Lorentzian(x, a, x0, sigma, y0):
    """Lorentzian peak

    Inputs:
    -------
        ``x``: independent variable
        ``a``: scaling factor (extremal value)
        ``x0``: center
        ``sigma``: half width at half maximum
        ``y0``: additive constant

    Formula:
    --------
        ``a/(1+((x-x0)/sigma)^2)+y0``
    """
    return a / (1 + ((x - x0) / sigma) ** 2) + y0