def fit_doublegauss(med,siglo,sighi,interval=0.683,p0=None,median=False,return_distribution=True):
    """Fits a two-sided Gaussian distribution to match a given confidence interval.

    The center of the distribution may be either the median or the mode.

    Parameters
    ----------
    med : float
        The center of the distribution to which to fit.  Default this
        will be the mode unless the `median` keyword is set to True.

    siglo : float
        Value at lower quantile (`q1 = 0.5 - interval/2`) to fit.  Often this is
        the "lower error bar."

    sighi : float
        Value at upper quantile (`q2 = 0.5 + interval/2`) to fit.  Often this is
        the "upper error bar."

    interval : float, optional
        The confidence interval enclosed by the provided error bars.  Default
        is 0.683 (1-sigma).

    p0 : array-like, optional
        Initial guess `doublegauss` parameters for the fit (`mu, sig1, sig2`).

    median : bool, optional
        Whether to treat the `med` parameter as the median or mode
        (default will be mode).

    return_distribution: bool, optional
        If `True`, then function will return a `DoubleGauss_Distribution` object.
        Otherwise, will return just the parameters.
    """
    if median:
        q1 = 0.5 - (interval/2)
        q2 = 0.5 + (interval/2)
        targetvals = np.array([med-siglo,med,med+sighi])
        qvals = np.array([q1,0.5,q2])
        def objfn(pars):
            logging.debug('{}'.format(pars))
            logging.debug('{} {}'.format(doublegauss_cdf(targetvals,pars),qvals))
            return doublegauss_cdf(targetvals,pars) - qvals

        if p0 is None:
            p0 = [med,siglo,sighi]
        pfit,success = leastsq(objfn,p0)

    else:
        q1 = 0.5 - (interval/2)
        q2 = 0.5 + (interval/2)
        targetvals = np.array([med-siglo,med+sighi])
        qvals = np.array([q1,q2])
        def objfn(pars):
            params = (med,pars[0],pars[1])
            return doublegauss_cdf(targetvals,params) - qvals

        if p0 is None:
            p0 = [siglo,sighi]
        pfit,success = leastsq(objfn,p0)
        pfit = (med,pfit[0],pfit[1])

    if return_distribution:
        dist = DoubleGauss_Distribution(*pfit)
        return dist
    else:
        return pfit