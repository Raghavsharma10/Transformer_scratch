def doublegauss_cdf(x,p):
    """Cumulative distribution function for two-sided Gaussian

    Parameters
    ----------
    x : float
        Input values at which to calculate CDF.

    p : array-like
        Parameters of distribution: (mu: mode of distribution,
                                     sig1: LH width,
                                     sig2: RH width)
    """
    x = np.atleast_1d(x)
    mu,sig1,sig2 = p
    sig1 = np.absolute(sig1)
    sig2 = np.absolute(sig2)
    ylo = float(sig1)/(sig1 + sig2)*(1 + erf((x-mu)/np.sqrt(2*sig1**2)))
    yhi = float(sig1)/(sig1 + sig2) + float(sig2)/(sig1+sig2)*(erf((x-mu)/np.sqrt(2*sig2**2)))
    lo = x < mu
    hi = x >= mu
    return ylo*lo + yhi*hi