def COM(z, M, **cosmo):
    """ Calculate concentration for halo of mass 'M' at redshift 'z'

    Parameters
    ----------
    z : float / numpy array
        Redshift to find concentration of halo
    M : float / numpy array
        Halo mass at redshift 'z'. Must be same size as 'z'
    cosmo : dict
        Dictionary of cosmological parameters, similar in format to:
        {'N_nu': 0,'Y_He': 0.24, 'h': 0.702, 'n': 0.963,'omega_M_0': 0.275,
         'omega_b_0': 0.0458,'omega_lambda_0': 0.725,'omega_n_0': 0.0,
         'sigma_8': 0.816, 't_0': 13.76, 'tau': 0.088,'z_reion': 10.6}

    Returns
    -------
    (c_array, sig_array, nu_array, zf_array) : float / numpy arrays
        of equivalent size to 'z' and 'M'. Variables are
        Concentration, Mass Variance 'sigma' this corresponds too,
        the dimnesionless fluctuation this represents and formation redshift

    """
    # Check that z and M are arrays
    z = np.array(z, ndmin=1, dtype=float)
    M = np.array(M, ndmin=1, dtype=float)

    # Create array
    c_array = np.empty_like(z)
    sig_array = np.empty_like(z)
    nu_array = np.empty_like(z)
    zf_array = np.empty_like(z)

    for i_ind, (zval, Mval) in enumerate(_izip(z, M)):
        # Evaluate the indices at each redshift and mass combination
        # that you want a concentration for, different to MAH which
        # uses one a_tilde and b_tilde at the starting redshift only
        a_tilde, b_tilde = calc_ab(zval, Mval, **cosmo)

        # Minimize equation to solve for 1 unknown, 'c'
        c = scipy.optimize.brentq(_minimize_c, 2, 1000,
                                  args=(zval, a_tilde, b_tilde,
                                        cosmo['A_scaling'], cosmo['omega_M_0'],
                                        cosmo['omega_lambda_0']))

        if np.isclose(c, 0):
            print("Error solving for concentration with given redshift and "
                  "(probably) too small a mass")
            c = -1
            sig = -1
            nu = -1
            zf = -1
        else:
            # Calculate formation redshift for this concentration,
            # redshift at which the scale radius = virial radius: z_-2
            zf = formationz(c, zval, Ascaling=cosmo['A_scaling'],
                            omega_M_0=cosmo['omega_M_0'],
                            omega_lambda_0=cosmo['omega_lambda_0'])

            R_Mass = cp.perturbation.mass_to_radius(Mval, **cosmo)

            sig, err_sig = cp.perturbation.sigma_r(R_Mass, 0, **cosmo)
            nu = 1.686/(sig*growthfactor(zval, norm=True, **cosmo))

        c_array[i_ind] = c
        sig_array[i_ind] = sig
        nu_array[i_ind] = nu
        zf_array[i_ind] = zf

    return(c_array, sig_array, nu_array, zf_array)