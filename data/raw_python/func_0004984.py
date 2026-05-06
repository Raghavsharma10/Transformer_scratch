def GuinierPorodGuinier(q, G, Rg1, alpha, Rg2):
    """Empirical Guinier-Porod-Guinier scattering

    Inputs:
    -------
        ``q``: independent variable
        ``G``: factor for the first Guinier-branch
        ``Rg1``: the first radius of gyration
        ``alpha``: the power-law exponent
        ``Rg2``: the second radius of gyration

    Formula:
    --------
        ``G*exp(-q^2*Rg1^2/3)`` if ``q<q_sep1``.
        ``A*q^alpha`` if ``q_sep1 <= q  <=q_sep2``.
        ``G2*exp(-q^2*Rg2^2/3)`` if ``q_sep2<q``.
        The parameters ``A``,``G2``, ``q_sep1``, ``q_sep2`` are determined
        from conditions of smoothness at the cross-overs.

    Literature:
    -----------
        B. Hammouda: A new Guinier-Porod model. J. Appl. Crystallogr. (2010) 43,
            716-719.

    """
    return GuinierPorodMulti(q, G, Rg1, alpha, Rg2)