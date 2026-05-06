def growthfactor(z, norm=True, **cosmo):
    """ Returns linear growth factor at a given redshift, normalised to z=0
        by default, for a given cosmology

    Parameters
    ----------

    z : float or numpy array
        The redshift at which the growth factor should be calculated
    norm : boolean, optional
        If true then normalise the growth factor to z=0 case defaults True
    cosmo : dict
        Dictionary of cosmological parameters, similar in format to:
        {'N_nu': 0,'Y_He': 0.24, 'h': 0.702, 'n': 0.963,'omega_M_0': 0.275,
         'omega_b_0': 0.0458,'omega_lambda_0': 0.725,'omega_n_0': 0.0,
         'sigma_8': 0.816, 't_0': 13.76, 'tau': 0.088,'z_reion': 10.6}

    Returns
    -------
    float or numpy array
        The growth factor at a range of redshifts 'z'

    Raises
    ------

    """
    H = np.sqrt(cosmo['omega_M_0'] * (1 + z)**3 +
                cosmo['omega_lambda_0'])
    growthval = H * _int_growth(z, **cosmo)
    if norm:
        growthval /= _int_growth(0, **cosmo)

    return(growthval)