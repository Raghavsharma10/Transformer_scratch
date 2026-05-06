def ExcludedVolumeChain(q, Rg, nu):
    """Scattering intensity of a generalized excluded-volume Gaussian chain

    Inputs:
    -------
        ``q``: independent variable
        ``Rg``: radius of gyration
        ``nu``: excluded volume exponent

    Formula:
    --------
        ``(u^(1/nu)*gamma(0.5/nu)*gammainc_lower(0.5/nu,u)-
            gamma(1/nu)*gammainc_lower(1/nu,u)) / (nu*u^(1/nu))``
        where ``u = q^2*Rg^2*(2*nu+1)*(2*nu+2)/6`` is the reduced scattering
        variable, ``gamma(x)`` is the gamma function and ``gammainc_lower(x,t)``
        is the lower incomplete gamma function.

    Literature:
    -----------
        SASFit manual 6. nov. 2010. Equation (3.60b)
    """
    u = (q * Rg) ** 2 * (2 * nu + 1) * (2 * nu + 2) / 6.
    return (u ** (0.5 / nu) * gamma(0.5 / nu) * gammainc(0.5 / nu, u) -
            gamma(1. / nu) * gammainc(1. / nu, u)) / (nu * u ** (1. / nu))