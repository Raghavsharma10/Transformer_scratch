def _delta_sigma(**cosmo):
    """ Perturb best-fit constant of proportionality Ascaling for
        rho_crit - rho_2 relation for unknown cosmology (Correa et al 2015c)

    Parameters
    ----------
    cosmo : dict
        Dictionary of cosmological parameters, similar in format to:
        {'N_nu': 0,'Y_He': 0.24, 'h': 0.702, 'n': 0.963,'omega_M_0': 0.275,
         'omega_b_0': 0.0458,'omega_lambda_0': 0.725,'omega_n_0': 0.0,
         'sigma_8': 0.816, 't_0': 13.76, 'tau': 0.088,'z_reion': 10.6}

    Returns
    -------
    float
        The perturbed 'A' relation between rho_2 and rho_crit for the cosmology

    Raises
    ------

    """

    M8_cosmo = cp.perturbation.radius_to_mass(8, **cosmo)
    perturbed_A = (0.796/cosmo['sigma_8']) * \
                  (M8_cosmo/2.5e14)**((cosmo['n']-0.963)/6)
    return(perturbed_A)