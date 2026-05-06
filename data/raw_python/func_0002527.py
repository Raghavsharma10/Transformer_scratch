def kld(p, q):
    """Kullback-Leiber divergence of two probability distributions pandas
    dataframes, p and q

    Parameters
    ----------
    p : pandas.DataFrame
        An nbins x features DataFrame, or (nbins,) Series
    q : pandas.DataFrame
        An nbins x features DataFrame, or (nbins,) Series

    Returns
    -------
    kld : pandas.Series
        Kullback-Lieber divergence of the common columns between the
        dataframe. E.g. between 1st column in p and 1st column in q, and 2nd
        column in p and 2nd column in q.

    Raises
    ------
    ValueError
        If the data provided is not a probability distribution, i.e. it has
        negative values or its columns do not sum to 1, raise ValueError

    Notes
    -----
    The input to this function must be probability distributions, not raw
    values. Otherwise, the output makes no sense.
    """
    try:
        _check_prob_dist(p)
        _check_prob_dist(q)
    except ValueError:
        return np.nan
    # If one of them is zero, then the other should be considered to be 0.
    # In this problem formulation, log0 = 0
    p = p.replace(0, np.nan)
    q = q.replace(0, np.nan)

    return (np.log2(p / q) * p).sum(axis=0)