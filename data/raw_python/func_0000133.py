def _deriv_growth(z, **cosmo):
    """ Returns derivative of the linear growth factor at z
        for a given cosmology **cosmo """

    inv_h = (cosmo['omega_M_0']*(1 + z)**3 + cosmo['omega_lambda_0'])**(-0.5)
    fz = (1 + z) * inv_h**3

    deriv_g = growthfactor(z, norm=True, **cosmo)*(inv_h**2) *\
        1.5 * cosmo['omega_M_0'] * (1 + z)**2 -\
        fz * growthfactor(z, norm=True, **cosmo)/_int_growth(z, **cosmo)

    return(deriv_g)