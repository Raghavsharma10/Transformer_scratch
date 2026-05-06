def double_lorgauss(x,p):
    """Evaluates a normalized distribution that is a mixture of a double-sided Gaussian and Double-sided Lorentzian.

    Parameters
    ----------
    x : float or array-like
        Value(s) at which to evaluate distribution

    p : array-like
        Input parameters: mu (mode of distribution),
                          sig1 (LH Gaussian width),
                          sig2 (RH Gaussian width),
                          gam1 (LH Lorentzian width),
                          gam2 (RH Lorentzian width),
                          G1 (LH Gaussian "strength"),
                          G2 (RH Gaussian "strength").

    Returns
    -------
    values : float or array-like
        Double LorGauss distribution evaluated at input(s).  If single value provided,
        single value returned. 
    """
    mu,sig1,sig2,gam1,gam2,G1,G2 = p
    gam1 = float(gam1)
    gam2 = float(gam2)

    G1 = abs(G1)
    G2 = abs(G2)
    sig1 = abs(sig1)
    sig2 = abs(sig2)
    gam1 = abs(gam1)
    gab2 = abs(gam2)
    
    L2 = (gam1/(gam1 + gam2)) * ((gam2*np.pi*G1)/(sig1*np.sqrt(2*np.pi)) - 
                                 (gam2*np.pi*G2)/(sig2*np.sqrt(2*np.pi)) +
                                 (gam2/gam1)*(4-G1-G2))
    L1 = 4 - G1 - G2 - L2

    
    #print G1,G2,L1,L2
    
    y1 = G1/(sig1*np.sqrt(2*np.pi)) * np.exp(-0.5*(x-mu)**2/sig1**2) +\
      L1/(np.pi*gam1) * gam1**2/((x-mu)**2 + gam1**2)
    y2 = G2/(sig2*np.sqrt(2*np.pi)) * np.exp(-0.5*(x-mu)**2/sig2**2) +\
      L2/(np.pi*gam2) * gam2**2/((x-mu)**2 + gam2**2)
    lo = (x < mu)
    hi = (x >= mu)
        
    return  y1*lo + y2*hi