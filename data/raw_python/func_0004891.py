def Cosine(x, a, omega, phi, y0):
    """Cosine function

    Inputs:
    -------
        ``x``: independent variable
        ``a``: amplitude
        ``omega``: circular frequency
        ``phi``: phase
        ``y0``: offset

    Formula:
    --------
        ``a*cos(x*omega + phi)+y0``
    """
    return a * np.cos(x * omega + phi) + y0