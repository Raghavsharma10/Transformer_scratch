def Sine(x, a, omega, phi, y0):
    """Sine function

    Inputs:
    -------
        ``x``: independent variable
        ``a``: amplitude
        ``omega``: circular frequency
        ``phi``: phase
        ``y0``: offset

    Formula:
    --------
        ``a*sin(x*omega + phi)+y0``
    """
    return a * np.sin(x * omega + phi) + y0