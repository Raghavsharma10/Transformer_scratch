def doublegauss(x,p):
    """Evaluates normalized two-sided Gaussian distribution

    Parameters
    ----------
    x : float or array-like
        Value(s) at which to evaluate distribution

    p : array-like
        Parameters of distribution: (mu: mode of distribution,
                                     sig1: LH width,
                                     sig2: RH width)

    Returns
    -------
    value : float or array-like
        Distribution evaluated at input value(s).  If single value provided,
        single value returned.
    """
    mu,sig1,sig2 = p
    x = np.atleast_1d(x)
    A = 1./(np.sqrt(2*np.pi)*(sig1+sig2)/2.)
    ylo = A*np.exp(-(x-mu)**2/(2*sig1**2))
    yhi = A*np.exp(-(x-mu)**2/(2*sig2**2))
    y = x*0
    wlo = np.where(x < mu)
    whi = np.where(x >= mu)
    y[wlo] = ylo[wlo]
    y[whi] = yhi[whi]
    if np.size(x)==1:
        return y[0]
    else:
        return y