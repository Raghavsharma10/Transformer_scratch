def getAscaling(cosmology, newcosmo=None):
    """ Returns the normalisation constant between
        Rho_-2 and Rho_mean(z_formation) for a given cosmology

    Parameters
    ----------
    cosmology : str or dict
        Can be named cosmology, default WMAP7 (aka DRAGONS), or
        DRAGONS, WMAP1, WMAP3, WMAP5, WMAP7, WMAP9, Planck13, Planck15
        or dictionary similar in format to:
        {'N_nu': 0,'Y_He': 0.24, 'h': 0.702, 'n': 0.963,'omega_M_0': 0.275,
         'omega_b_0': 0.0458,'omega_lambda_0': 0.725,'omega_n_0': 0.0,
         'sigma_8': 0.816, 't_0': 13.76, 'tau': 0.088,'z_reion': 10.6}
    newcosmo : str, optional
        If cosmology is not from predefined list have to perturbation
        A_scaling variable. Defaults to None.

    Returns
    -------
    float
        The scaled 'A' relation between rho_2 and rho_crit for the cosmology

    """
    # Values from Correa 15c
    defaultcosmologies = {'dragons': 887, 'wmap1': 853, 'wmap3': 850,
                          'wmap5': 887, 'wmap7': 887, 'wmap9': 950,
                          'wmap1_lss': 853, 'wmap3_mean': 850,
                          'wmap5_ml': 887, 'wmap5_lss': 887,
                          'wmap7_lss': 887,
                          'planck13': 880, 'planck15': 880}

    if newcosmo:
        # Scale from default WMAP5 cosmology using Correa et al 14b eqn C1
        A_scaling = defaultcosmologies['wmap5'] * _delta_sigma(**cosmology)
    else:
        if cosmology.lower() in defaultcosmologies.keys():
            A_scaling = defaultcosmologies[cosmology.lower()]
        else:
            print("Error, don't recognise your cosmology for A_scaling ")
            print("You provided %s" % (cosmology))

    return(A_scaling)