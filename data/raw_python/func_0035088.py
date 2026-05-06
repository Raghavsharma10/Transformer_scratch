def c_DuttonMaccio(z, m, h=h):
    """Concentration from c(M) relation in Dutton & Maccio (2014).

    Parameters
    ----------
    z : float or array_like
        Redshift(s) of halos.
    m : float or array_like
        Mass(es) of halos (m200 definition), in units of solar masses.
    h : float, optional
        Hubble parameter. Default is from Planck13.

    Returns
    ----------
    ndarray
        Concentration values (c200) for halos.

    References
    ----------
    Calculation from Planck-based results of simulations presented in:

    A.A. Dutton & A.V. Maccio, "Cold dark matter haloes in the Planck era:
    evolution of structural parameters for Einasto and NFW profiles,"
    Monthly Notices of the Royal Astronomical Society, Volume 441, Issue 4,
    p.3359-3374, 2014.
    """

    z, m = _check_inputs(z, m)

    a = 0.52 + 0.385 * np.exp(-0.617 * (z**1.21))  # EQ 10
    b = -0.101 + 0.026 * z                         # EQ 11

    logc200 = a + b * np.log10(m * h / (10.**12))  # EQ 7

    concentration = 10.**logc200

    return concentration