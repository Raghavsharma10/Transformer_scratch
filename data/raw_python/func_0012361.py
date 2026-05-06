def bootstrap_resample_run(ns_run, threads=None, ninit_sep=False,
                           random_seed=False):
    """Bootstrap resamples threads of nested sampling run, returning a new
    (resampled) nested sampling run.

    Get the individual threads for a nested sampling run.

    Parameters
    ----------
    ns_run: dict
        Nested sampling run dictionary.
    threads: None or list of numpy arrays, optional
    ninit_sep: bool
        For dynamic runs: resample initial threads and dynamically added
        threads separately. Useful when there are only a few threads which
        start by sampling the whole prior, as errors occur if none of these are
        included in the bootstrap resample.
    random_seed: None, bool or int, optional
        Set numpy random seed. Default is to use None (so a random seed is
        chosen from the computer's internal state) to ensure reliable results
        when multiprocessing. Can set to an integer or to False to not edit the
        seed.


    Returns
    -------
    ns_run_temp: dict
        Nested sampling run dictionary.
    """
    if random_seed is not False:
        # save the random state so we don't affect other bits of the code
        state = np.random.get_state()
        np.random.seed(random_seed)
    if threads is None:
        threads = nestcheck.ns_run_utils.get_run_threads(ns_run)
    n_threads = len(threads)
    if ninit_sep:
        try:
            ninit = ns_run['settings']['ninit']
            assert np.all(ns_run['thread_min_max'][:ninit, 0] == -np.inf), (
                'ninit_sep assumes the initial threads are labeled '
                '(0,...,ninit-1), so these should start by sampling the whole '
                'prior.')
            inds = np.random.randint(0, ninit, ninit)
            inds = np.append(inds, np.random.randint(ninit, n_threads,
                                                     n_threads - ninit))
        except KeyError:
            warnings.warn((
                'bootstrap_resample_run has kwarg ninit_sep=True but '
                'ns_run["settings"]["ninit"] does not exist. Doing bootstrap '
                'with ninit_sep=False'), UserWarning)
            ninit_sep = False
    if not ninit_sep:
        inds = np.random.randint(0, n_threads, n_threads)
    threads_temp = [threads[i] for i in inds]
    resampled_run = nestcheck.ns_run_utils.combine_threads(threads_temp)
    try:
        resampled_run['settings'] = ns_run['settings']
    except KeyError:
        pass
    if random_seed is not False:
        # if we have used a random seed then return to the original state
        np.random.set_state(state)
    return resampled_run