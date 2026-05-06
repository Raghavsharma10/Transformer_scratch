def get_run_threads(ns_run):
    """
    Get the individual threads from a nested sampling run.

    Parameters
    ----------
    ns_run: dict
        Nested sampling run dict (see data_processing module docstring for more
        details).

    Returns
    -------
    threads: list of numpy array
        Each thread (list element) is a samples array containing columns
        [logl, thread label, change in nlive at sample, (thetas)]
        with each row representing a single sample.
    """
    samples = array_given_run(ns_run)
    unique_threads = np.unique(ns_run['thread_labels'])
    assert ns_run['thread_min_max'].shape[0] == unique_threads.shape[0], (
        'some threads have no points! {0} != {1}'.format(
            unique_threads.shape[0], ns_run['thread_min_max'].shape[0]))
    threads = []
    for i, th_lab in enumerate(unique_threads):
        thread_array = samples[np.where(samples[:, 1] == th_lab)]
        # delete changes in nlive due to other threads in the run
        thread_array[:, 2] = 0
        thread_array[-1, 2] = -1
        min_max = np.reshape(ns_run['thread_min_max'][i, :], (1, 2))
        assert min_max[0, 1] == thread_array[-1, 0], (
            'thread max logl should equal logl of its final point!')
        threads.append(dict_given_run_array(thread_array, min_max))
    return threads