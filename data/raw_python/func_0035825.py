def split_array_like(df, columns=None): #TODO rename TODO if it's not a big performance hit, just make them arraylike? We already indicated the column explicitly (sort of) so...
    '''
    Split cells with array-like values along row axis.

    Column names are maintained. The index is dropped.

    Parameters
    ----------
    df : ~pandas.DataFrame
        Data frame ``df[columns]`` should contain :py:class:`~pytil.numpy.ArrayLike`
        values.
    columns : ~typing.Collection[str] or str or None
        Columns (or column) whose values to split. Defaults to ``df.columns``.

    Returns
    -------
    ~pandas.DataFrame
        Data frame with array-like values in ``df[columns]`` split across rows,
        and corresponding values in other columns repeated.

    Examples
    --------
    >>> df = pd.DataFrame([[1,[1,2],[1]],[1,[1,2],[3,4,5]],[2,[1],[1,2]]], columns=('check', 'a', 'b'))
    >>> df
       check       a          b
    0      1  [1, 2]        [1]
    1      1  [1, 2]  [3, 4, 5]
    2      2     [1]     [1, 2]
    >>> split_array_like(df, ['a', 'b'])
      check  a  b
    0     1  1  1
    1     1  2  1
    2     1  1  3
    3     1  1  4
    4     1  1  5
    5     1  2  3
    6     1  2  4
    7     1  2  5
    8     2  1  1
    9     2  1  2
    '''
    # TODO could add option to keep_index by using reset_index and eventually
    # set_index. index names trickery: MultiIndex.names, Index.name. Both can be
    # None. If Index.name can be None in which case it translates to 'index' or
    # if that already exists, it translates to 'level_0'. If MultiIndex.names is
    # None, it translates to level_0,... level_N
    dtypes = df.dtypes

    if columns is None:
        columns = df.columns
    elif isinstance(columns, str):
        columns = [columns]

    for column in columns:
        expanded = np.repeat(df.values, df[column].apply(len).values, axis=0)
        expanded[:, df.columns.get_loc(column)] = np.concatenate(df[column].tolist())
        df = pd.DataFrame(expanded, columns=df.columns)

    # keep types unchanged
    for i, dtype in enumerate(dtypes):
        df.iloc[:,i] = df.iloc[:,i].astype(dtype)

    return df