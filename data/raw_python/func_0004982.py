def PorodGuinier(q, a, alpha, Rg):
    """Empirical Porod-Guinier scattering

    Inputs:
    -------
        ``q``: independent variable
        ``a``: factor of the power-law branch
        ``alpha``: power-law exponent
        ``Rg``: radius of gyration

    Formula:
    --------
        ``G * exp(-q^2*Rg^2/3)`` if ``q>q_sep`` and ``a*q^alpha`` otherwise.
        ``q_sep`` and ``G`` are determined from conditions of smoothness at
        the cross-over.

    Literature:
    -----------
        B. Hammouda: A new Guinier-Porod model. J. Appl. Crystallogr. (2010) 43,
            716-719.
    """
    return PorodGuinierMulti(q, a, alpha, Rg)