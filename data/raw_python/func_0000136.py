def formationz(c, z, Ascaling=900, omega_M_0=0.25, omega_lambda_0=0.75):
    """ Rearrange eqn 18 from Correa et al (2015c) to return
        formation redshift for a concentration at a given redshift

    Parameters
    ----------
    c : float / numpy array
        Concentration of halo
    z : float / numpy array
        Redshift of halo with concentration c
    Ascaling : float
        Cosmological dependent scaling between densities, use function
        getAscaling('WMAP5') if unsure. Default is 900.
    omega_M_0 : float
        Mass density of the universe. Default is 0.25
    omega_lambda_0 : float
        Dark Energy density of the universe. Default is 0.75

    Returns
    -------
    zf : float / numpy array
        Formation redshift for halo of concentration 'c' at redshift 'z'

    """
    Y1 = np.log(2) - 0.5
    Yc = np.log(1+c) - c/(1+c)
    rho_2 = 200*(c**3)*Y1/Yc

    zf = (((1+z)**3 + omega_lambda_0/omega_M_0) *
          (rho_2/Ascaling) - omega_lambda_0/omega_M_0)**(1/3) - 1

    return(zf)