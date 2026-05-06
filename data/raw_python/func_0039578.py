def sin_fisher(y,k):
    """pdf for y=sin(x) if x is fisher-distributed with parameter k.  Support is [0,1).
    """
    return 1/np.sqrt(1-y**2) * (k/(np.sinh(k)) * y * (np.cosh(k*np.sqrt(1-y**2))))