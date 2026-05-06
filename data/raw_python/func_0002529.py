def entropy(binned, base=2):
    """Find the entropy of each column of a dataframe

    Parameters
    ----------
    binned : pandas.DataFrame
        A nbins x features DataFrame of probability distributions, where each
        column sums to 1
    base : numeric
        The log-base of the entropy. Default is 2, so the resulting entropy
        is in bits.

    Returns
    -------
    entropy : pandas.Seires
        Entropy values for each column of the dataframe.

    Raises
    ------
    ValueError
        If the data provided is not a probability distribution, i.e. it has
        negative values or its columns do not sum to 1, raise ValueError
    """
    try:
        _check_prob_dist(binned)
    except ValueError:
        np.nan
    return -((np.log(binned) / np.log(base)) * binned).sum(axis=0)