def get_dummy_thread(nsamples, **kwargs):
    """Generate dummy data for a single nested sampling thread.

    Log-likelihood values of points are generated from a uniform distribution
    in (0, 1), sorted, scaled by logl_range and shifted by logl_start (if it is
    not -np.inf). Theta values of each point are each generated from a uniform
    distribution in (0, 1).

    Parameters
    ----------
    nsamples: int
        Number of samples in thread.
    ndim: int, optional
        Number of dimensions.
    seed: int, optional
        If not False, the seed is set with np.random.seed(seed).
    logl_start: float, optional
        logl at which thread starts.
    logl_range: float, optional
        Scale factor applied to logl values.
    """
    seed = kwargs.pop('seed', False)
    ndim = kwargs.pop('ndim', 2)
    logl_start = kwargs.pop('logl_start', -np.inf)
    logl_range = kwargs.pop('logl_range', 1)
    if kwargs:
        raise TypeError('Unexpected **kwargs: {0}'.format(kwargs))
    if seed is not False:
        np.random.seed(seed)
    thread = {'logl': np.sort(np.random.random(nsamples)) * logl_range,
              'nlive_array': np.full(nsamples, 1.),
              'theta': np.random.random((nsamples, ndim)),
              'thread_labels': np.zeros(nsamples).astype(int)}
    if logl_start != -np.inf:
        thread['logl'] += logl_start
    thread['thread_min_max'] = np.asarray([[logl_start, thread['logl'][-1]]])
    return thread