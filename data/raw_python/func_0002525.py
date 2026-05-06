def bin_range_strings(bins, fmt=':g'):
    """Given a list of bins, make a list of strings of those bin ranges

    Parameters
    ----------
    bins : list_like
        List of anything, usually values of bin edges

    Returns
    -------
    bin_ranges : list
        List of bin ranges

    >>> bin_range_strings((0, 0.5, 1))
    ['0-0.5', '0.5-1']
    """
    return [('{' + fmt + '}-{' + fmt + '}').format(i, j)
            for i, j in zip(bins, bins[1:])]