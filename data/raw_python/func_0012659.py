def fit2d(samples,e_x, e_y, remove_zeros = False, p_est = None,  **kw):
    """Fits a 2D distribution with splines.

    Input:
        samples: Matrix or list of arrays 
            If matrix, it must be of size Nx2, where N is the number of
            observations. If list, it must contain two arrays of length
            N.
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
    if p_est is None:
        height = len(e_y)-1
        width = len(e_x)-1   
        (p_est, _) = np.histogramdd(samples, (e_x, e_y))
    else:
        p_est = p_est.T
        width, height = p_est.shape
    # p_est contains x in dim 1 and y in dim 0
    shape = p_est.shape
    p_est = (p_est/sum(p_est.flat)).reshape(shape)
    mx =  p_est.sum(1)
    my = p_est.sum(0)
    # Transpose hist to have x in dim 0
    p_est = p_est.T.flatten()
    basis, knots = spline_base2d(width, height, marginal_x = mx, marginal_y = my, **kw)
    model = linear_model.BayesianRidge()
    if remove_zeros:
        non_zero = ~(p_est == 0)
        model.fit(basis[:, non_zero].T, p_est[non_zero])
    else:
        non_zero = (p_est >= 0)
        p_est[~non_zero,:] = np.finfo(float).eps
        model.fit(basis.T, p_est)
    return (model.predict(basis.T).reshape((height, width)), 
            p_est.reshape((height, width)), knots)