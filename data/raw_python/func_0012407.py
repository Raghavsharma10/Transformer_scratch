def dict_given_run_array(samples, thread_min_max):
    """
    Converts an array of information about samples back into a nested sampling
    run dictionary (see data_processing module docstring for more details).

    N.B. the output dict only contains the following keys: 'logl',
    'thread_label', 'nlive_array', 'theta'. Any other keys giving additional
    information about the run output cannot be reproduced from the function
    arguments, and are therefore ommitted.

    Parameters
    ----------
    samples: numpy array
        Numpy array containing columns
        [logl, thread label, change in nlive at sample, (thetas)]
        with each row representing a single sample.
    thread_min_max': numpy array, optional
        2d array with a row for each thread containing the likelihoods at which
        it begins and ends.
        Needed to calculate nlive_array (otherwise this is set to None).

    Returns
    -------
    ns_run: dict
        Nested sampling run dict (see data_processing module docstring for more
        details).
    """
    ns_run = {'logl': samples[:, 0],
              'thread_labels': samples[:, 1],
              'thread_min_max': thread_min_max,
              'theta': samples[:, 3:]}
    if np.all(~np.isnan(ns_run['thread_labels'])):
        ns_run['thread_labels'] = ns_run['thread_labels'].astype(int)
        assert np.array_equal(samples[:, 1], ns_run['thread_labels']), ((
            'Casting thread labels from samples array to int has changed '
            'their values!\nsamples[:, 1]={}\nthread_labels={}').format(
                samples[:, 1], ns_run['thread_labels']))
    nlive_0 = (thread_min_max[:, 0] <= ns_run['logl'].min()).sum()
    assert nlive_0 > 0, 'nlive_0={}'.format(nlive_0)
    nlive_array = np.zeros(samples.shape[0]) + nlive_0
    nlive_array[1:] += np.cumsum(samples[:-1, 2])
    # Check if there are multiple threads starting on the first logl point
    dup_th_starts = (thread_min_max[:, 0] == ns_run['logl'].min()).sum()
    if dup_th_starts > 1:
        # In this case we approximate the true nlive (which we dont really
        # know) by making sure the array's final point is 1 and setting all
        # points with logl = logl.min() to have the same nlive
        nlive_array += (1 - nlive_array[-1])
        n_logl_min = (ns_run['logl'] == ns_run['logl'].min()).sum()
        nlive_array[:n_logl_min] = nlive_0
        warnings.warn((
            'duplicate starting logls: {} threads start at logl.min()={}, '
            'and {} points have logl=logl.min(). nlive_array may only be '
            'approximately correct.').format(
                dup_th_starts, ns_run['logl'].min(), n_logl_min), UserWarning)
    assert nlive_array.min() > 0, ((
        'nlive contains 0s or negative values. nlive_0={}'
        '\nnlive_array = {}\nthread_min_max={}').format(
            nlive_0, nlive_array, thread_min_max))
    assert nlive_array[-1] == 1, (
        'final point in nlive_array != 1.\nnlive_array = ' + str(nlive_array))
    ns_run['nlive_array'] = nlive_array
    return ns_run