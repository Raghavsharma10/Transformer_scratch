def min_between(min_reps=3, max_reps=8, percentile=0.33):
    """Function to decide the minimum number of reps to perform
    given `min_reps` and `max_rep`.

    Parameters
    ----------
    min_reps
        The minimum number of repeitions.

    max_reps
        The maximum number of repetitions.

    percentile
        The percentile to cap at.

    Return
    -------
    (low, high)
        A tuple containing a new rep range.


    Examples
    -------
    >>> min_between(min_reps = 3, max_reps = 8, percentile = 0.33)
    (3, 5)
    """
    higher_limit = min_reps + (max_reps - min_reps) * percentile
    return min_reps, math.ceil(higher_limit)