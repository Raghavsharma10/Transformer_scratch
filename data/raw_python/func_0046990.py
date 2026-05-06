def identify_groups(groups):
    """
    To compute number of unique elements in a given membership specification.

    Parameters
    ----------
    groups : numpy 1d array of length p, each value specifying which group that particular node belongs to.
        For examlpe, if you have a cortical thickness values for 1000 vertices belonging to 100 patches,
        this array could  have numbers 1 to 100 specifying which vertex belongs to which cortical patch.
        Although grouping with numerical values (contiguous from 1 to num_patches) is strongly recommended for simplicity,
        this could also be a list of strings of length p.

    Returns
    -------
    group_ids : numpy array of values identifying the unique groups specified
    num_groups : scalar value denoting the number of unique groups specified

    """

    group_ids = np.unique(groups)
    num_groups = len(group_ids)

    if num_groups < 2:
        raise ValueError('There must be atleast two nodes or groups in data, for pair-wise edge-weight calculations.')

    return group_ids, num_groups