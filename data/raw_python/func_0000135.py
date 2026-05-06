def _minimize_c(c, z=0, a_tilde=1, b_tilde=-1,
                Ascaling=900, omega_M_0=0.25, omega_lambda_0=0.75):
    """ Trial function to solve 2 eqns (17 and 18) from Correa et al. (2015c)
        for 1 unknown, i.e. concentration, returned by a minimisation call """

    # Fn 1 (LHS of Eqn 18)

    Y1 = np.log(2) - 0.5
    Yc = np.log(1+c) - c/(1+c)
    f1 = Y1/Yc

    # Fn 2 (RHS of Eqn 18)

    # Eqn 14 - Define the mean inner density
    rho_2 = 200 * c**3 * Y1 / Yc

    # Eqn 17 rearranged to solve for Formation Redshift
    # essentially when universe had rho_2 density
    zf = (((1 + z)**3 + omega_lambda_0/omega_M_0) *
          (rho_2/Ascaling) - omega_lambda_0/omega_M_0)**(1/3) - 1

    # RHS of Eqn 19
    f2 = ((1 + zf - z)**a_tilde) * np.exp((zf - z) * b_tilde)

    # LHS - RHS should be zero for the correct concentration
    return(f1-f2)