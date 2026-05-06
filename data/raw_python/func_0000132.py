def _int_growth(z, **cosmo):
    """ Returns integral of the linear growth factor from z=200 to z=z """

    zmax = 200

    if hasattr(z, "__len__"):
        for zval in z:
            assert(zval < zmax)
    else:
        assert(z < zmax)

    y, yerr = scipy.integrate.quad(
        lambda z: (1 + z)/(cosmo['omega_M_0']*(1 + z)**3 +
                           cosmo['omega_lambda_0'])**(1.5),
        z, zmax)

    return(y)