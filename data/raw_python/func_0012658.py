def fit3d(samples, e_x, e_y, e_z, remove_zeros = False, **kw):
    """Fits a 3D distribution with splines.

    Input:
        samples: Array
            Array of samples from a probability distribution
        e_x: Array
            Edges that define the events in the probability 
            distribution along the x direction. For example, 
            e_x[0] < samples[0] <= e_x[1] picks out all 
            samples that are associated with the first event.
        e_y: Array
            See e_x, but for the y direction.
        remove_zeros: Bool
            If True, events that are not observed will not 
            be part of the fitting process. If False, those 
            events will be modelled as finfo('float').eps 
        **kw: Arguments that are passed on to spline_bse1d.

    Returns:
        distribution: Array
            An array that gives an estimate of probability for 
            events defined by e.
        knots: Tuple of arrays
            Sequence of knots that were used for the spline basis (x,y) 
    """
    height, width, depth = len(e_y)-1, len(e_x)-1, len(e_z)-1 
    
    (p_est, _) = np.histogramdd(samples, (e_x, e_y, e_z))
    p_est = p_est/sum(p_est.flat)
    p_est = p_est.flatten()
    if remove_zeros:
        non_zero = ~(p_est == 0)
    else:
        non_zero = (p_est >= 0)
    basis = spline_base3d(width,height, depth, **kw)
    model = linear_model.BayesianRidge()
    model.fit(basis[:, non_zero].T, p_est[:,np.newaxis][non_zero,:])
    return (model.predict(basis.T).reshape((width, height, depth)), 
                p_est.reshape((width, height, depth)))