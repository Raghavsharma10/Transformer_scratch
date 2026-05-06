def equals(df1, df2, ignore_order=set(), ignore_indices=set(), all_close=False, _return_reason=False):
    '''
    Get whether 2 data frames are equal.

    ``NaN`` is considered equal to ``NaN`` and `None`.

    Parameters
    ----------
    df1 : ~pandas.DataFrame
        Data frame to compare.
    df2 : ~pandas.DataFrame
        Data frame to compare.
    ignore_order : ~typing.Set[int]
        Axi in which to ignore order.
    ignore_indices : ~typing.Set[int]
        Axi of which to ignore the index. E.g. ``{1}`` allows differences in
        ``df.columns.name`` and ``df.columns.equals(df2.columns)``.
    all_close : bool
        If `False`, values must match exactly, if `True`, floats are compared as if
        compared with `numpy.isclose`.
    _return_reason : bool
        Internal. If `True`, `equals` returns a tuple containing the reason, else
        `equals` only returns a bool indicating equality (or equivalence
        rather).

    Returns
    -------
    bool
        Whether they are equal (after ignoring according to the parameters).

        Internal note: if ``_return_reason``, ``Tuple[bool, str or None]`` is
        returned. The former is whether they're equal, the latter is `None` if
        equal or a short explanation of why the data frames aren't equal,
        otherwise.

    Notes
    -----
    All values (including those of indices) must be copyable and ``__eq__`` must
    be such that a copy must equal its original. A value must equal itself
    unless it's ``NaN``. Values needn't be orderable or hashable (however
    pandas requires index values to be orderable and hashable). By consequence,
    this is not an efficient function, but it is flexible.

    Examples
    --------
    >>> from pytil import data_frame as df_
    >>> import pandas as pd
    >>> df = pd.DataFrame([
    ...        [1, 2, 3],
    ...        [4, 5, 6],
    ...        [7, 8, 9]
    ...    ],
    ...    index=pd.Index(('i1', 'i2', 'i3'), name='index1'),
    ...    columns=pd.Index(('c1', 'c2', 'c3'), name='columns1')
    ... )
    >>> df
    columns1  c1  c2  c3
    index1              
    i1         1   2   3
    i2         4   5   6
    i3         7   8   9
    >>> df2 = df.reindex(('i3', 'i1', 'i2'), columns=('c2', 'c1', 'c3'))
    >>> df2
    columns1  c2  c1  c3
    index1              
    i3         8   7   9
    i1         2   1   3
    i2         5   4   6
    >>> df_.equals(df, df2)
    False
    >>> df_.equals(df, df2, ignore_order=(0,1))
    True
    >>> df2 = df.copy()
    >>> df2.index = [1,2,3]
    >>> df2
    columns1  c1  c2  c3
    1          1   2   3
    2          4   5   6
    3          7   8   9
    >>> df_.equals(df, df2)
    False
    >>> df_.equals(df, df2, ignore_indices={0})
    True
    >>> df2 = df.reindex(('i3', 'i1', 'i2'))
    >>> df2
    columns1  c1  c2  c3
    index1              
    i3         7   8   9
    i1         1   2   3
    i2         4   5   6
    >>> df_.equals(df, df2, ignore_indices={0})  # does not ignore row order!
    False
    >>> df_.equals(df, df2, ignore_order={0})
    True
    >>> df2 = df.copy()
    >>> df2.index.name = 'other'
    >>> df_.equals(df, df2)  # df.index.name must match as well, same goes for df.columns.name
    False
    '''
    result = _equals(df1, df2, ignore_order, ignore_indices, all_close)
    if _return_reason:
        return result
    else:
        return result[0]