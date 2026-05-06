def GeneralGuinier(q, G, Rg, s):
    """Generalized Guinier scattering

    Inputs:
    -------
        ``q``: independent variable
        ``G``: factor
        ``Rg``: radius of gyration
        ``s``: dimensionality parameter (can be 1, 2, 3)

    Formula:
    --------
        ``G/q**(3-s)*exp(-(q^2*Rg^2)/s)``
    """
    return G / q ** (3 - s) * np.exp(-(q * Rg) ** 2 / s)