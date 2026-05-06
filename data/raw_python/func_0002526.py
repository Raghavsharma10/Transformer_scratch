def binify(data, bins):
    """Makes a histogram of each column the provided binsize

    Parameters
    ----------
    data : pandas.DataFrame
        A samples x features dataframe. Each feature (column) will be binned
        into the provided bins
    bins : iterable
        Bins you would like to use for this data. Must include the final bin
        value, e.g. (0, 0.5, 1) for the two bins (0, 0.5) and (0.5, 1).
        nbins = len(bins) - 1

    Returns
    -------
    binned : pandas.DataFrame
        An nbins x features DataFrame of each column binned across rows
    """
    if bins is None:
        raise ValueError('Must specify "bins"')
    if isinstance(data, pd.DataFrame):
        binned = data.apply(lambda x: pd.Series(np.histogram(x, bins=bins,
                                                             range=(0, 1))[0]))
    elif isinstance(data, pd.Series):
        binned = pd.Series(np.histogram(data, bins=bins, range=(0, 1))[0])
    else:
        raise ValueError('`data` must be either a 1d vector or 2d matrix')
    binned.index = bin_range_strings(bins)

    # Normalize so each column sums to 1
    binned = binned / binned.sum().astype(float)
    return binned