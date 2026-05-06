def split(series):
    '''
    Split values.

    The index is dropped, but this may change in the future.

    Parameters
    ----------
    series : ~pandas.Series[~pytil.numpy.ArrayLike]
        Series with array-like values.

    Returns
    -------
    ~pandas.Series
        Series with values split across rows.

    Examples
    --------
    >>> series = pd.Series([[1,2],[1,2],[3,4,5]])
    >>> series
    0       [1, 2]
    1       [1, 2]
    2    [3, 4, 5]
    dtype: object
    >>> split(series)
    0    1
    1    2
    2    1
    3    2
    4    3
    5    4
    6    5
    dtype: object
    '''
    s = df_.split_array_like(series.apply(list).to_frame('column'), 'column')['column']
    s.name = series.name
    return s