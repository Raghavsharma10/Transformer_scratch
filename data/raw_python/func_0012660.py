def fit1d(samples, e, remove_zeros = False, **kw):
    """Fits a 1D distribution with splines.

    Input:
        samples: Array
            Array of samples from a probability distribution
        e: Array
            Edges that define the events in the probability 
            distribution. For example, e[0] < x <= e[1] is
            the range of values that are associated with the
            first event.
        **kw: Arguments that are passed on to spline_bse1d.

    Returns:
        distribution: Array
            An array that gives an estimate of probability for 
            events defined by e.
        knots: Array
            Sequence of knots that were used for the spline basis
    """
    samples = samples[~np.isnan(samples)]
    length = len(e)-1
    hist,_ = np.histogramdd(samples, (e,))
    hist = hist/sum(hist)
    basis, knots = spline_base1d(length, marginal = hist, **kw)
    non_zero = hist>0
    model = linear_model.BayesianRidge()
    if remove_zeros:
        model.fit(basis[non_zero, :], hist[:,np.newaxis][non_zero,:])
    else:
        hist[~non_zero] = np.finfo(float).eps
        model.fit(basis, hist[:,np.newaxis])
    return model.predict(basis), hist, knots