def invert(series):
    '''
    Swap index with values of series.

    Parameters
    ----------
    series : ~pandas.Series
        Series to swap on, must have a name.

    Returns
    -------
    ~pandas.Series
        Series after swap.

    See also
    --------
    pandas.Series.map
        Joins series ``a -> b`` and ``b -> c`` into ``a -> c``.
    '''
    df = series.reset_index() #TODO alt is to to_frame and then use som dataframe methods
    df.set_index(series.name, inplace=True)
    return df[df.columns[0]]