def diff_means(array_one, array_two):
    """
    Computes the difference in means between two arrays of values.

    Given arrays will be flattened (to 1D array) regardless of dimension,
        and any non-finite/NaN values will be ignored.

    Parameters
    ----------
    array_one, array_two : iterable
        Two arrays of values, possibly of different length.

    Returns
    -------
    diff_medians : float
        scalar measuring the difference in medians, ignoring NaNs/non-finite values.

    Raises
    ------
    ValueError
        If one or more of the arrays are empty.

    """

    array_one = check_array(array_one)
    array_two = check_array(array_two)
    diff_means = np.ma.mean(array_one) - np.ma.mean(array_two)

    return diff_means