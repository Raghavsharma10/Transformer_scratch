def check_ns_run_members(run):
    """Check nested sampling run member keys and values.

    Parameters
    ----------
    run: dict
        nested sampling run to check.

    Raises
    ------
    AssertionError
        if run does not have expected properties.
    """
    run_keys = list(run.keys())
    # Mandatory keys
    for key in ['logl', 'nlive_array', 'theta', 'thread_labels',
                'thread_min_max']:
        assert key in run_keys
        run_keys.remove(key)
    # Optional keys
    for key in ['output']:
        try:
            run_keys.remove(key)
        except ValueError:
            pass
    # Check for unexpected keys
    assert not run_keys, 'Unexpected keys in ns_run: ' + str(run_keys)
    # Check type of mandatory members
    for key in ['logl', 'nlive_array', 'theta', 'thread_labels',
                'thread_min_max']:
        assert isinstance(run[key], np.ndarray), (
            key + ' is type ' + type(run[key]).__name__)
    # check shapes of keys
    assert run['logl'].ndim == 1
    assert run['logl'].shape == run['nlive_array'].shape
    assert run['logl'].shape == run['thread_labels'].shape
    assert run['theta'].ndim == 2
    assert run['logl'].shape[0] == run['theta'].shape[0]