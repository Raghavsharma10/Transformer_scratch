def PowerlawGuinierPorodConst(q, A, alpha, G, Rg, beta, C):
    """Sum of a Power-law, a Guinier-Porod curve and a constant.

    Inputs:
    -------
        ``q``: independent variable (momentum transfer)
        ``A``: scaling factor of the power-law
        ``alpha``: power-law exponent
        ``G``: scaling factor of the Guinier-Porod curve
        ``Rg``: Radius of gyration
        ``beta``: power-law exponent of the Guinier-Porod curve
        ``C``: additive constant

    Formula:
    --------
        ``A*q^alpha + GuinierPorod(q,G,Rg,beta) + C``
    """
    return PowerlawPlusConstant(q, A, alpha, C) + GuinierPorod(q, G, Rg, beta)