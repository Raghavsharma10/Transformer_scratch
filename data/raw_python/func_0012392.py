def threads_given_birth_inds(birth_inds):
    """Divides a nested sampling run into threads, using info on the indexes
    at which points were sampled. See "Sampling errors in nested sampling
    parameter estimation" (Higson et al. 2018) for more information.

    Parameters
    ----------
    birth_inds: 1d numpy array
        Indexes of the iso-likelihood contours from within which each point was
        sampled ("born").

    Returns
    -------
    thread_labels: 1d numpy array of ints
        labels of the thread each point belongs to.
    """
    unique, counts = np.unique(birth_inds, return_counts=True)
    # First get a list of all the indexes on which threads start and their
    # counts. This is every point initially sampled from the prior, plus any
    # indexes where more than one point is sampled.
    thread_start_inds = np.concatenate((
        unique[:1], unique[1:][counts[1:] > 1]))
    thread_start_counts = np.concatenate((
        counts[:1], counts[1:][counts[1:] > 1] - 1))
    thread_labels = np.full(birth_inds.shape, np.nan)
    thread_num = 0
    for nmulti, multi in enumerate(thread_start_inds):
        for i, start_ind in enumerate(np.where(birth_inds == multi)[0]):
            # unless nmulti=0 the first point born on the contour (i=0) is
            # already assigned to a thread
            if i != 0 or nmulti == 0:
                # check point has not already been assigned
                assert np.isnan(thread_labels[start_ind])
                thread_labels[start_ind] = thread_num
                # find the point which replaced it
                next_ind = np.where(birth_inds == start_ind)[0]
                while next_ind.shape != (0,):
                    # check point has not already been assigned
                    assert np.isnan(thread_labels[next_ind[0]])
                    thread_labels[next_ind[0]] = thread_num
                    # find the point which replaced it
                    next_ind = np.where(birth_inds == next_ind[0])[0]
                thread_num += 1
    if not np.all(~np.isnan(thread_labels)):
        warnings.warn((
            '{} points (out of a total of {}) were not given a thread label! '
            'This is likely due to small numerical errors in your nested '
            'sampling software while running the calculation or writing the '
            'input files. '
            'I will try to give an approximate answer by randomly assigning '
            'these points to threads.'
            '\nIndexes without labels are {}'
            '\nIndexes on which threads start are {} with {} threads '
            'starting on each.').format(
                (np.isnan(thread_labels)).sum(), birth_inds.shape[0],
                np.where(np.isnan(thread_labels))[0],
                thread_start_inds, thread_start_counts))
        inds = np.where(np.isnan(thread_labels))[0]
        state = np.random.get_state()  # Save random state before seeding
        np.random.seed(0)  # make thread decomposition is reproducible
        for ind in inds:
            # Get the set of threads with members both before and after ind to
            # ensure we don't change nlive_array by extending a thread
            labels_to_choose = np.intersect1d(  # N.B. this removes nans too
                thread_labels[:ind], thread_labels[ind + 1:])
            if labels_to_choose.shape[0] == 0:
                # In edge case that there is no intersection, just randomly
                # select from non-nan thread labels
                labels_to_choose = np.unique(
                    thread_labels[~np.isnan(thread_labels)])
            thread_labels[ind] = np.random.choice(labels_to_choose)
        np.random.set_state(state)  # Reset random state
    assert np.all(~np.isnan(thread_labels)), (
        '{} points still do not have thread labels'.format(
            (np.isnan(thread_labels)).sum()))
    assert np.array_equal(thread_labels, thread_labels.astype(int)), (
        'Thread labels should all be ints!')
    thread_labels = thread_labels.astype(int)
    # Check unique thread labels are a sequence from 0 to nthreads-1
    assert np.array_equal(
        np.unique(thread_labels),
        np.asarray(range(sum(thread_start_counts)))), (
            str(np.unique(thread_labels)) + ' is not equal to range('
            + str(sum(thread_start_counts)) + ')')
    return thread_labels