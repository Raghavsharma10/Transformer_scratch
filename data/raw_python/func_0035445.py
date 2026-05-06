def covar_rescaling_factor_efficient(C):
    """
    Returns the rescaling factor for the Gower normalizion on covariance matrix C
    the rescaled covariance matrix has sample variance of 1
    """
    n = C.shape[0]
    P = sp.eye(n) - sp.ones((n,n))/float(n)
    CP = C - C.mean(0)[:, sp.newaxis]
    trPCP = sp.sum(P * CP)
    r = (n-1) / trPCP
    return r