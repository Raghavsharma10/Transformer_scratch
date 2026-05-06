def calc_ab(zi, Mi, **cosmo):
    """ Calculate growth rate indices a_tilde and b_tilde

    Parameters
    ----------
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
    (a_tilde, b_tilde) : float
    """

    # When zi = 0, the a_tilde becomes alpha and b_tilde becomes beta

    # Eqn 23 of Correa et al 2015a (analytically solve from Eqn 16 and 17)
    # Arbitray formation redshift, z_-2 in COM is more physically motivated
    zf = -0.0064 * (np.log10(Mi))**2 + 0.0237 * (np.log10(Mi)) + 1.8837

    # Eqn 22 of Correa et al 2015a
    q = 4.137 * zf**(-0.9476)

    # Radius of a mass Mi
    R_Mass = cp.perturbation.mass_to_radius(Mi, **cosmo)  # [Mpc]
    # Radius of a mass Mi/q
    Rq_Mass = cp.perturbation.mass_to_radius(Mi/q, **cosmo)  # [Mpc]

    # Mass variance 'sigma' evaluate at z=0 to a good approximation
    sig, err_sig = cp.perturbation.sigma_r(R_Mass, 0, **cosmo)  # [Mpc]
    sigq, err_sigq = cp.perturbation.sigma_r(Rq_Mass, 0, **cosmo)  # [Mpc]

    f = (sigq**2 - sig**2)**(-0.5)

    # Eqn 9 and 10 from Correa et al 2015c
    # (generalised to zi from Correa et al 2015a's z=0 special case)
    # a_tilde is power law growth rate
    a_tilde = (np.sqrt(2/np.pi) * 1.686 * _deriv_growth(zi, **cosmo) /
               growthfactor(zi, norm=True, **cosmo)**2 + 1)*f
    # b_tilde is exponential growth rate
    b_tilde = -f

    return(a_tilde, b_tilde)