def Gaussian(x, a, x0, sigma, y0):
    """Gaussian peak

    Inputs:
    -------
        ``x``: independent variable
        ``a``: scaling factor (extremal value)
        ``x0``: center
        ``sigma``: half width at half maximum
        ``y0``: additive constant

    Formula:
    --------
        ``a*exp(-(x-x0)^2)/(2*sigma^2)+y0``
    """
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2)) + y0