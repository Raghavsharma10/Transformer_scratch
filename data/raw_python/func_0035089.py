def c_Duffy(z, m, h=h):
    """Concentration from c(M) relation published in Duffy et al. (2008).

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
    Results from N-body simulations using WMAP5 cosmology, presented in:

    A.R. Duffy, J. Schaye, S.T. Kay, and C. Dalla Vecchia, "Dark matter
    halo concentrations in the Wilkinson Microwave Anisotropy Probe year 5
    cosmology," Monthly Notices of the Royal Astronomical Society, Volume
    390, Issue 1, pp. L64-L68, 2008.

    This calculation uses the parameters corresponding to the NFW model,
    the '200' halo definition, and the 'full' sample of halos spanning
    z = 0-2. This means the values of fitted parameters (A,B,C) = (5.71,
    -0.084,-0.47) in Table 1 of Duffy et al. (2008).
    """

    z, m = _check_inputs(z, m)

    M_pivot = 2.e12 / h  # [M_solar]

    A = 5.71
    B = -0.084
    C = -0.47

    concentration = A * ((m / M_pivot)**B) * (1 + z)**C

    return concentration