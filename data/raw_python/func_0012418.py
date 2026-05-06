def check_ns_run_threads(run):
    """Check thread labels and thread_min_max have expected properties.

    Parameters
    ----------
    run: dict
        Nested sampling run to check.

    Raises
    ------
    AssertionError
        If run does not have expected properties.
    """
    assert run['thread_labels'].dtype == int
    uniq_th = np.unique(run['thread_labels'])
    assert np.array_equal(
        np.asarray(range(run['thread_min_max'].shape[0])), uniq_th), \
        str(uniq_th)
    # Check thread_min_max
    assert np.any(run['thread_min_max'][:, 0] == -np.inf), (
        'Run should have at least one thread which starts by sampling the ' +
        'whole prior')
    for th_lab in uniq_th:
        inds = np.where(run['thread_labels'] == th_lab)[0]
        th_info = 'thread label={}, first_logl={}, thread_min_max={}'.format(
            th_lab, run['logl'][inds[0]], run['thread_min_max'][th_lab, :])
        assert run['thread_min_max'][th_lab, 0] <= run['logl'][inds[0]], (
            'First point in thread has logl less than thread min logl! ' +
            th_info + ', difference={}'.format(
                run['logl'][inds[0]] - run['thread_min_max'][th_lab, 0]))
        assert run['thread_min_max'][th_lab, 1] == run['logl'][inds[-1]], (
            'Last point in thread logl != thread end logl! ' + th_info)