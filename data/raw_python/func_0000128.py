def _getcosmoheader(cosmo):
    """ Output the cosmology to a string for writing to file """

    cosmoheader = ("# Cosmology (flat) Om:{0:.3f}, Ol:{1:.3f}, h:{2:.2f}, "
                   "sigma8:{3:.3f}, ns:{4:.2f}".format(
                       cosmo['omega_M_0'], cosmo['omega_lambda_0'], cosmo['h'],
                       cosmo['sigma_8'], cosmo['n']))

    return(cosmoheader)