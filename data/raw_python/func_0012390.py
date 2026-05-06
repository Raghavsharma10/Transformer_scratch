def birth_inds_given_contours(birth_logl_arr, logl_arr, **kwargs):
    """Maps the iso-likelihood contours on which points were born to the
    index of the dead point on this contour.

    MultiNest and PolyChord use different values to identify the inital live
    points which were sampled from the whole prior (PolyChord uses -1e+30
    and MultiNest -0.179769313486231571E+309). However in each case the first
    dead point must have been sampled from the whole prior, so for either
    package we can use

    init_birth = birth_logl_arr[0]

    If there are many points with the same logl_arr and dup_assert is False,
    these points are randomly assigned an order (to ensure results are
    consistent, random seeding is used).

    Parameters
    ----------
    logl_arr: 1d numpy array
        logl values of each point.
    birth_logl_arr: 1d numpy array
        Birth contours - i.e. logl values of the iso-likelihood contour from
        within each point was sampled (on which it was born).
    dup_assert: bool, optional
        See ns_run_utils.check_ns_run_logls docstring.
    dup_warn: bool, optional
        See ns_run_utils.check_ns_run_logls docstring.

    Returns
    -------
    birth_inds: 1d numpy array of ints
        Step at which each element of logl_arr was sampled. Points sampled from
        the whole prior are assigned value -1.
    """
    dup_assert = kwargs.pop('dup_assert', False)
    dup_warn = kwargs.pop('dup_warn', False)
    if kwargs:
        raise TypeError('Unexpected **kwargs: {0}'.format(kwargs))
    assert logl_arr.ndim == 1, logl_arr.ndim
    assert birth_logl_arr.ndim == 1, birth_logl_arr.ndim
    # Check for duplicate logl values (if specified by dup_assert or dup_warn)
    nestcheck.ns_run_utils.check_ns_run_logls(
        {'logl': logl_arr}, dup_assert=dup_assert, dup_warn=dup_warn)
    # Random seed so results are consistent if there are duplicate logls
    state = np.random.get_state()  # Save random state before seeding
    np.random.seed(0)
    # Calculate birth inds
    init_birth = birth_logl_arr[0]
    assert np.all(birth_logl_arr <= logl_arr), (
        logl_arr[birth_logl_arr > logl_arr])
    birth_inds = np.full(birth_logl_arr.shape, np.nan)
    birth_inds[birth_logl_arr == init_birth] = -1
    for i, birth_logl in enumerate(birth_logl_arr):
        if not np.isnan(birth_inds[i]):
            # birth ind has already been assigned
            continue
        dup_deaths = np.where(logl_arr == birth_logl)[0]
        if dup_deaths.shape == (1,):
            # death index is unique
            birth_inds[i] = dup_deaths[0]
            continue
        # The remainder of this loop deals with the case that multiple points
        # have the same logl value (=birth_logl). This can occur due to limited
        # precision, or for likelihoods with contant regions. In this case we
        # randomly assign the duplicates birth steps in a manner
        # that provides a valid division into nested sampling runs
        dup_births = np.where(birth_logl_arr == birth_logl)[0]
        assert dup_deaths.shape[0] > 1, dup_deaths
        if np.all(birth_logl_arr[dup_deaths] != birth_logl):
            # If no points both are born and die on this contour, we can just
            # randomly assign an order
            np.random.shuffle(dup_deaths)
            inds_to_use = dup_deaths
        else:
            # If some points are both born and die on the contour, we need to
            # take care that the assigned birth inds do not result in some
            # points dying before they are born
            try:
                inds_to_use = sample_less_than_condition(
                    dup_deaths, dup_births)
            except ValueError:
                raise ValueError((
                    'There is no way to allocate indexes dup_deaths={} such '
                    'that each is less than dup_births={}.').format(
                        dup_deaths, dup_births))
        try:
            # Add our selected inds_to_use values to the birth_inds array
            # Note that dup_deaths (and hence inds to use) may have more
            # members than dup_births, because one of the duplicates may be
            # the final point in a thread. We therefore include only the first
            # dup_births.shape[0] elements
            birth_inds[dup_births] = inds_to_use[:dup_births.shape[0]]
        except ValueError:
            warnings.warn((
                'for logl={}, the number of points born (indexes='
                '{}) is bigger than the number of points dying '
                '(indexes={}). This indicates a problem with your '
                'nested sampling software - it may be caused by '
                'a bug in PolyChord which was fixed in PolyChord '
                'v1.14, so try upgrading. I will try to give an '
                'approximate allocation of threads but this may '
                'fail.').format(
                    birth_logl, dup_births, inds_to_use), UserWarning)
            extra_inds = np.random.choice(
                inds_to_use, size=dup_births.shape[0] - inds_to_use.shape[0])
            inds_to_use = np.concatenate((inds_to_use, extra_inds))
            np.random.shuffle(inds_to_use)
            birth_inds[dup_births] = inds_to_use[:dup_births.shape[0]]
    assert np.all(~np.isnan(birth_inds)), np.isnan(birth_inds).sum()
    np.random.set_state(state)  # Reset random state
    return birth_inds.astype(int)