def DRAGONS(flat=False, extras=True):
    """DRAGONS cosmology assumes WMAP7 + BAO + H_0 mean from
    Komatsu et al. (2011) ApJS 192 18K (arxiv:1001.4538v1)

    Parameters
    ----------

    flat: boolean

      If True, sets omega_lambda_0 = 1 - omega_M_0 to ensure omega_k_0
      = 0 exactly. Also sets omega_k_0 = 0 explicitly.

    extras: boolean

      If True, sets neutrino number N_nu = 0, neutrino density
      omega_n_0 = 0.0, Helium mass fraction Y_He = 0.24.

      """
    omega_c_0 = 0.2292
    omega_b_0 = 0.0458
    cosmo = {'omega_b_0': omega_b_0,
             'omega_M_0': omega_b_0 + omega_c_0,
             'omega_lambda_0': 0.725,
             'h': 0.702,
             'n': 0.963,
             'sigma_8': 0.816,
             'tau': 0.088,
             'z_reion': 10.6,
             't_0': 13.76,
             }
    if flat:
        cosmo['omega_lambda_0'] = 1 - cosmo['omega_M_0']
        cosmo['omega_k_0'] = 0.0
    if extras:
        add_extras(cosmo)
    return cosmo