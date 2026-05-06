def rel_posterior_mass(logx, logl):
    """Calculate the relative posterior mass for some array of logx values
    given the likelihood, prior and number of dimensions.
    The posterior mass at each logX value is proportional to L(X)X, where L(X)
    is the likelihood.
    The weight is returned normalized so that the integral of the weight with
    respect to logX is 1.

    Parameters
    ----------
    logx: 1d numpy array
        Logx values at which to calculate posterior mass.
    logl: 1d numpy array
        Logl values corresponding to each logx (same shape as logx).

    Returns
    -------
    w_rel: 1d numpy array
        Relative posterior mass at each input logx value.
    """
    logw = logx + logl
    w_rel = np.exp(logw - logw.max())
    w_rel /= np.abs(np.trapz(w_rel, x=logx))
    return w_rel