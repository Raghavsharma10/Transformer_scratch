def DebyeChain(q, Rg):
    """Scattering form-factor intensity of a Gaussian chain (Debye)

    Inputs:
    -------
        ``q``: independent variable
        ``Rg``: radius of gyration

    Formula:
    --------
        ``2*(exp(-a)-1+a)/a^2`` where ``a=(q*Rg)^2``
    """
    a = (q * Rg) ** 2
    return 2 * (np.exp(-a) - 1 + a) / a ** 2