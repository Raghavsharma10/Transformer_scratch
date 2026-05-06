def covar_rescaling_factor(C):
    """
    Returns the rescaling factor for the Gower normalizion on covariance matrix C
    the rescaled covariance matrix has sample variance of 1
    """
    n = C.shape[0]
    P = sp.eye(n) - sp.ones((n,n))/float(n)
    trPCP = sp.trace(sp.dot(P,sp.dot(C,P)))
    r = (n-1) / trPCP
    return r