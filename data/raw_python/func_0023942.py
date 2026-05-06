def parallel_compute_ll_matrix(gp, bounds, num_pts, num_proc=None):
    """Compute matrix of the log likelihood over the parameter space in parallel.
    
    Parameters
    ----------
    bounds : 2-tuple or list of 2-tuples with length equal to the number of free parameters
        Bounds on the range to use for each of the parameters. If a single
        2-tuple is given, it will be used for each of the parameters.
    num_pts : int or list of ints with length equal to the number of free parameters
        The number of points to use for each parameters. If a single int is
        given, it will be used for each of the parameters.
    num_proc : Positive int or None, optional
        Number of processes to run the parallel computation with. If set to
        None, ALL available cores are used. Default is None (use all available
        cores).
    
    Returns
    -------
    ll_vals : array
        The log likelihood for each of the parameter possibilities.
    param_vals : list of array
        The parameter values used.
    """
    if num_proc is None:
        num_proc = multiprocessing.cpu_count()
    
    present_free_params = gp.free_params
    
    bounds = scipy.atleast_2d(scipy.asarray(bounds, dtype=float))
    if bounds.shape[1] != 2:
        raise ValueError("Argument bounds must have shape (n, 2)!")
    # If bounds is a single tuple, repeat it for each free parameter:
    if bounds.shape[0] == 1:
        bounds = scipy.tile(bounds, (len(present_free_params), 1))
    # If num_pts is a single value, use it for all of the parameters:
    try:
        iter(num_pts)
    except TypeError:
        num_pts = num_pts * scipy.ones(bounds.shape[0], dtype=int)
    else:
        num_pts = scipy.asarray(num_pts, dtype=int)
        if len(num_pts) != len(present_free_params):
            raise ValueError("Length of num_pts must match the number of free parameters of kernel!")
    
    # Form arrays to evaluate parameters over:
    param_vals = []
    for k in xrange(0, len(present_free_params)):
        param_vals.append(scipy.linspace(bounds[k, 0], bounds[k, 1], num_pts[k]))
    
    pv_cases = list()
    gp_cases = list()
    num_pts_cases = list()
    for k in xrange(0, len(param_vals[0])):
        specific_param_vals = list(param_vals)
        specific_param_vals[0] = param_vals[0][k]
        pv_cases.append(specific_param_vals)
        
        gp_cases += [copy.deepcopy(gp)]
        
        num_pts_cases.append(num_pts)
    
    pool =  multiprocessing.Pool(processes=num_proc)    
    try:
        vals = scipy.asarray(
            pool.map(
                _compute_ll_matrix_wrapper,
                zip(gp_cases, pv_cases, num_pts_cases)
            )
        )
    finally:
        pool.close()
    
    return (vals, param_vals)