def Planck_2015(flat=False, extras=True):
    """Planck 2015 XII: Cosmological parameters Table 4
    column Planck TT, TE, EE + lowP + lensing + ext
    from Ade et al. (2015) A&A in press (arxiv:1502.01589v1)

    Parameters
    ----------

    flat: boolean

      If True, sets omega_lambda_0 = 1 - omega_M_0 to ensure omega_k_0
      = 0 exactly. Also sets omega_k_0 = 0 explicitly.

    extras: boolean

      If True, sets neutrino number N_nu = 0, neutrino density
      omega_n_0 = 0.0, Helium mass fraction Y_He = 0.24.

      """
    omega_b_0 = 0.02230/(0.6774**2)
    cosmo = {'omega_b_0': omega_b_0,
             'omega_M_0': 0.3089,
             'omega_lambda_0': 0.6911,
             'h': 0.6774,
             'n': 0.9667,
             'sigma_8': 0.8159,
             'tau': 0.066,
             'z_reion': 8.8,
             't_0': 13.799,
             }
    if flat:
        cosmo['omega_lambda_0'] = 1 - cosmo['omega_M_0']
        cosmo['omega_k_0'] = 0.0
    if extras:
        add_extras(cosmo)
    return cosmo