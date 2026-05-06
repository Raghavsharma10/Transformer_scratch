def combine_threads(threads, assert_birth_point=False):
    """
    Combine list of threads into a single ns run.
    This is different to combining runs as repeated threads are allowed, and as
    some threads can start from log-likelihood contours on which no dead
    point in the run is present.

    Note that if all the thread labels are not unique and in ascending order,
    the output will fail check_ns_run. However provided the thread labels are
    not used it will work ok for calculations based on nlive, logl and theta.

    Parameters
    ----------
    threads: list of dicts
        List of nested sampling run dicts, each representing a single thread.
    assert_birth_point: bool, optional
        Whether or not to assert there is exactly one point present in the run
        with the log-likelihood at which each point was born. This is not true
        for bootstrap resamples of runs, where birth points may be repeated or
        not present at all.

    Returns
    -------
    run: dict
        Nested sampling run dict (see data_processing module docstring for more
        details).
    """
    thread_min_max = np.vstack([td['thread_min_max'] for td in threads])
    assert len(threads) == thread_min_max.shape[0]
    # construct samples array from the threads, including an updated nlive
    samples_temp = np.vstack([array_given_run(thread) for thread in threads])
    samples_temp = samples_temp[np.argsort(samples_temp[:, 0])]
    # update the changes in live points column for threads which start part way
    # through the run. These are only present in dynamic nested sampling.
    logl_starts = thread_min_max[:, 0]
    state = np.random.get_state()  # save random state
    np.random.seed(0)  # seed to make sure any random assignment is repoducable
    for logl_start in logl_starts[logl_starts != -np.inf]:
        ind = np.where(samples_temp[:, 0] == logl_start)[0]
        if assert_birth_point:
            assert ind.shape == (1,), \
                'No unique birth point! ' + str(ind.shape)
        if ind.shape == (1,):
            # If the point at which this thread started is present exactly
            # once in this bootstrap replication:
            samples_temp[ind[0], 2] += 1
        elif ind.shape == (0,):
            # If the point with the likelihood at which the thread started
            # is not present in this particular bootstrap replication,
            # approximate it with the point with the nearest likelihood.
            ind_closest = np.argmin(np.abs(samples_temp[:, 0] - logl_start))
            samples_temp[ind_closest, 2] += 1
        else:
            # If the point at which this thread started is present multiple
            # times in this bootstrap replication, select one at random to
            # increment nlive on. This avoids any systematic bias from e.g.
            # always choosing the first point.
            samples_temp[np.random.choice(ind), 2] += 1
    np.random.set_state(state)
    # make run
    ns_run = dict_given_run_array(samples_temp, thread_min_max)
    try:
        check_ns_run_threads(ns_run)
    except AssertionError:
        # If the threads are not valid (e.g. for bootstrap resamples) then
        # set them to None so they can't be accidentally used
        ns_run['thread_labels'] = None
        ns_run['thread_min_max'] = None
    return ns_run