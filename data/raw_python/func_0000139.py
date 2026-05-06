def MAH(z, zi, Mi, **cosmo):
    """ Calculate mass accretion history by looping function acc_rate
        over redshift steps 'z' for halo of mass 'Mi' at redshift 'zi'

    Parameters
    ----------
    z : float / numpy array
        Redshift to output MAH over. Note zi<z always
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
    (dMdt, Mz) : float / numpy arrays of equivalent size to 'z'
        Accretion rate [Msol/yr], halo mass [Msol] at redshift 'z'

    """

    # Ensure that z is a 1D NumPy array
    z = np.array(z, ndmin=1, dtype=float)

    # Create a full array
    dMdt_array = np.empty_like(z)
    Mz_array = np.empty_like(z)

    for i_ind, zval in enumerate(z):
        # Solve the accretion rate and halo mass at each redshift step
        dMdt, Mz = acc_rate(zval, zi, Mi, **cosmo)

        dMdt_array[i_ind] = dMdt
        Mz_array[i_ind] = Mz

    return(dMdt_array, Mz_array)