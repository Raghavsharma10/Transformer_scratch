def maxwellian(r, r0, n):
    """Maxwellian-like distribution of spherical particles
    
    Inputs:
    -------
        r: np.ndarray or scalar
            radii
        r0: positive scalar or ErrorValue
            mean radius
        n: positive scalar or ErrorValue
            "n" parameter
    
    Output:
    -------
        the distribution function and its uncertainty as an ErrorValue containing arrays.
        The uncertainty of 'r0' and 'n' is taken into account.
        
    Notes:
    ------
        M(r)=2*r^n/r0^(n+1)*exp(-r^2/r0^2) / gamma((n+1)/2)
    """
    r0 = ErrorValue(r0)
    n = ErrorValue(n)

    expterm = np.exp(-r ** 2 / r0.val ** 2)
    dmaxdr0 = -2 * r ** n.val * r0.val ** (-n.val - 4) * ((n.val + 1) * r0.val ** 2 - 2 * r ** 2) * expterm / gamma((n.val + 1) * 0.5)
    dmaxdn = -r ** n.val * r0.val ** (-n.val - 1) * expterm * (2 * np.log(r0.val) - 2 * np.log(r) + psi((n.val + 1) * 0.5)) / gamma((n.val + 1) * 0.5)

    maxwellian = 2 * r ** n.val * r0.val ** (-n.val - 1) * expterm / gamma((n.val + 1) * 0.5)
    dmaxwellian = (dmaxdn ** 2 * n.err ** 2 + dmaxdr0 ** 2 * r0.err ** 2) ** 0.5
    return ErrorValue(maxwellian, dmaxwellian)