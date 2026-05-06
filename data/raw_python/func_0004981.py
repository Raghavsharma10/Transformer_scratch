def GuinierPorod(q, G, Rg, alpha):
    """Empirical Guinier-Porod scattering

    Inputs:
    -------
        ``q``: independent variable
        ``G``: factor of the Guinier-branch
        ``Rg``: radius of gyration
        ``alpha``: power-law exponent

    Formula:
    --------
        ``G * exp(-q^2*Rg^2/3)`` if ``q<q_sep`` and ``a*q^alpha`` otherwise.
        ``q_sep`` and ``a`` are determined from conditions of smoothness at
        the cross-over.

    Literature:
    -----------
        B. Hammouda: A new Guinier-Porod model. J. Appl. Crystallogr. (2010) 43,
            716-719.
    """
    return GuinierPorodMulti(q, G, Rg, alpha)