def binify_and_jsd(df1, df2, bins, pair=None):
    """Binify and calculate jensen-shannon divergence between two dataframes

    Parameters
    ----------
    df1, df2 : pandas.DataFrames
        Dataframes to calculate JSD between columns of. Must have overlapping
        column names
    bins : array-like
        Bins to use for transforming df{1,2} into probability distributions
    pair : str, optional
        Name of the pair to save as the name of the series

    Returns
    -------
    divergence : pandas.Series
        The Jensen-Shannon divergence between columns of df1, df2
    """
    binned1 = binify(df1, bins=bins).dropna(how='all', axis=1)
    binned2 = binify(df2, bins=bins).dropna(how='all', axis=1)

    binned1, binned2 = binned1.align(binned2, axis=1, join='inner')

    series = np.sqrt(jsd(binned1, binned2))
    series.name = pair
    return series