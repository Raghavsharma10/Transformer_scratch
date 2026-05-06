def diff_medians_abs(array_one, array_two):
    """
    Computes the absolute (symmetric) difference in medians between two arrays of values.

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

    abs_diff_medians = np.abs(diff_medians(array_one, array_two))

    return abs_diff_medians