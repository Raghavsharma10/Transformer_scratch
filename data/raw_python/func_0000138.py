def acc_rate(z, zi, Mi, **cosmo):
    """ Calculate accretion rate and mass history of a halo at any
        redshift 'z' with mass 'Mi' at a lower redshift 'z'

    Parameters
    ----------
    z : float
        Redshift to solve acc_rate / mass history. Note zi<z
    zi : float
        Redshift
    Mi : float
        Halo mass at redshift 'zi'
    cosmo : dict
        Dictionary of cosmological parameters, similar in format to:
        {'N_nu': 0,'Y_He': 0.24, 'h': 0.702, 'n': 0.963,'omega_M_0': 0.275,
         'omega_b_0': 0.0458,'omega_lambda_0': 0.725,'omega_n_0': 0.0,
         'sigma_8': 0.816, 't_0': 13.76, 'tau': 0.088,'z_reion': 10.6}

    Returns
    -------
    (dMdt, Mz) : float
        Accretion rate [Msol/yr], halo mass [Msol] at redshift 'z'

    """
    # Find parameters a_tilde and b_tilde for initial redshift
    # use Eqn 9 and 10 of Correa et al. (2015c)
    a_tilde, b_tilde = calc_ab(zi, Mi, **cosmo)

    # Halo mass at z, in Msol
    # use Eqn 8 in Correa et al. (2015c)
    Mz = Mi * ((1 + z - zi)**a_tilde) * (np.exp(b_tilde * (z - zi)))

    # Accretion rate at z, Msol yr^-1
    # use Eqn 11 from Correa et al. (2015c)
    dMdt = 71.6 * (Mz/1e12) * (cosmo['h']/0.7) *\
        (-a_tilde / (1 + z - zi) - b_tilde) * (1 + z) *\
        np.sqrt(cosmo['omega_M_0']*(1 + z)**3+cosmo['omega_lambda_0'])

    return(dMdt, Mz)