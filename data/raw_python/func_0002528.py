def jsd(p, q):
    """Finds the per-column JSD between dataframes p and q

    Jensen-Shannon divergence of two probability distrubutions pandas
    dataframes, p and q. These distributions are usually created by running
    binify() on the dataframe.

    Parameters
    ----------
    p : pandas.DataFrame
        An nbins x features DataFrame.
    q : pandas.DataFrame
        An nbins x features DataFrame.

    Returns
    -------
    jsd : pandas.Series
        Jensen-Shannon divergence of each column with the same names between
        p and q

    Raises
    ------
    ValueError
        If the data provided is not a probability distribution, i.e. it has
        negative values or its columns do not sum to 1, raise ValueError
    """
    try:
        _check_prob_dist(p)
        _check_prob_dist(q)
    except ValueError:
        return np.nan
    weight = 0.5
    m = weight * (p + q)

    result = weight * kld(p, m) + (1 - weight) * kld(q, m)
    return result