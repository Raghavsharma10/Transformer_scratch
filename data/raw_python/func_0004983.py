def PorodGuinierPorod(q, a, alpha, Rg, beta):
    """Empirical Porod-Guinier-Porod scattering

    Inputs:
    -------
        ``q``: independent variable
        ``a``: factor of the first power-law branch
        ``alpha``: exponent of the first power-law branch
        ``Rg``: radius of gyration
        ``beta``: exponent of the second power-law branch

    Formula:
    --------
        ``a*q^alpha`` if ``q<q_sep1``. ``G * exp(-q^2*Rg^2/3)`` if
        ``q_sep1<q<q_sep2`` and ``b*q^beta`` if ``q_sep2<q``.
        ``q_sep1``, ``q_sep2``, ``G`` and ``b`` are determined from conditions
        of smoothness at the cross-overs.

    Literature:
    -----------
        B. Hammouda: A new Guinier-Porod model. J. Appl. Crystallogr. (2010) 43,
            716-719.
    """
    return PorodGuinierMulti(q, a, alpha, Rg, beta)