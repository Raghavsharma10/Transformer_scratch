def computational_form(data):
    """
    Input Series of numbers, Series, or DataFrames repackaged
    for calculation.

    Parameters
    ----------
    data : pandas.Series
        Series of numbers, Series, DataFrames

    Returns
    -------
    pandas.Series, DataFrame, or Panel
        repacked data, aligned by indices, ready for calculation
    """

    if isinstance(data.iloc[0], DataFrame):
        dslice = Panel.from_dict(dict([(i,data.iloc[i])
                                       for i in xrange(len(data))]))
    elif isinstance(data.iloc[0], Series):
        dslice = DataFrame(data.tolist())
        dslice.index = data.index
    else:
        dslice = data
    return dslice