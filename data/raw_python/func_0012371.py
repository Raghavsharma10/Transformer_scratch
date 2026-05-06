def get_dummy_run(nthread, nsamples, **kwargs):
    """Generate dummy data for a nested sampling run.

    Log-likelihood values of points are generated from a uniform distribution
    in (0, 1), sorted, scaled by logl_range and shifted by logl_start (if it is
    not -np.inf). Theta values of each point are each generated from a uniform
    distribution in (0, 1).

    Parameters
    ----------
    nthreads: int
        Number of threads in the run.
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
    threads = []
    # set seed before generating any threads and do not reset for each thread
    if seed is not False:
        np.random.seed(seed)
    threads = []
    for _ in range(nthread):
        threads.append(get_dummy_thread(
            nsamples, ndim=ndim, seed=False, logl_start=logl_start,
            logl_range=logl_range))
    # Sort threads in order of starting logl so labels match labels that would
    # have been given processing a dead points array. N.B. this only works when
    # all threads have same start_logl
    threads = sorted(threads, key=lambda th: th['logl'][0])
    for i, _ in enumerate(threads):
        threads[i]['thread_labels'] = np.full(nsamples, i)
    # Use combine_ns_runs rather than combine threads as this relabels the
    # threads according to their order
    return nestcheck.ns_run_utils.combine_threads(threads)