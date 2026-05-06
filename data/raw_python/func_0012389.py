def process_samples_array(samples, **kwargs):
    """Convert an array of nested sampling dead and live points of the type
    produced by PolyChord and MultiNest into a nestcheck nested sampling run
    dictionary.

    Parameters
    ----------
    samples: 2d numpy array
        Array of dead points and any remaining live points at termination.
        Has #parameters + 2 columns:
        param_1, param_2, ... , logl, birth_logl
    kwargs: dict, optional
        Options passed to birth_inds_given_contours

    Returns
    -------
    ns_run: dict
        Nested sampling run dict (see the module docstring for more
        details). Only contains information in samples (not additional
        optional output key).
    """
    samples = samples[np.argsort(samples[:, -2])]
    ns_run = {}
    ns_run['logl'] = samples[:, -2]
    ns_run['theta'] = samples[:, :-2]
    birth_contours = samples[:, -1]
    # birth_contours, ns_run['theta'] = check_logls_unique(
    #     samples[:, -2], samples[:, -1], samples[:, :-2])
    birth_inds = birth_inds_given_contours(
        birth_contours, ns_run['logl'], **kwargs)
    ns_run['thread_labels'] = threads_given_birth_inds(birth_inds)
    unique_threads = np.unique(ns_run['thread_labels'])
    assert np.array_equal(unique_threads,
                          np.asarray(range(unique_threads.shape[0])))
    # Work out nlive_array and thread_min_max logls from thread labels and
    # birth contours
    thread_min_max = np.zeros((unique_threads.shape[0], 2))
    # NB delta_nlive indexes are offset from points' indexes by 1 as we need an
    # element to represent the initial sampling of live points before any dead
    # points are created.
    # I.E. birth on step 1 corresponds to replacing dead point zero
    delta_nlive = np.zeros(samples.shape[0] + 1)
    for label in unique_threads:
        thread_inds = np.where(ns_run['thread_labels'] == label)[0]
        # Max is final logl in thread
        thread_min_max[label, 1] = ns_run['logl'][thread_inds[-1]]
        thread_start_birth_ind = birth_inds[thread_inds[0]]
        # delta nlive indexes are +1 from logl indexes to allow for initial
        # nlive (before first dead point)
        delta_nlive[thread_inds[-1] + 1] -= 1
        if thread_start_birth_ind == birth_inds[0]:
            # thread minimum is -inf as it starts by sampling from whole prior
            thread_min_max[label, 0] = -np.inf
            delta_nlive[0] += 1
        else:
            assert thread_start_birth_ind >= 0
            thread_min_max[label, 0] = ns_run['logl'][thread_start_birth_ind]
            delta_nlive[thread_start_birth_ind + 1] += 1
    ns_run['thread_min_max'] = thread_min_max
    ns_run['nlive_array'] = np.cumsum(delta_nlive)[:-1]
    return ns_run