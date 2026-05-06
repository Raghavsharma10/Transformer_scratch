def equals(series1, series2, ignore_order=False, ignore_index=False, all_close=False, _return_reason=False):
    '''
    Get whether 2 series are equal.

    ``NaN`` is considered equal to ``NaN`` and `None`.

    Parameters
    ----------
    series1 : pandas.Series
        Series to compare.
    series2 : pandas.Series
        Series to compare.
    ignore_order : bool
        Ignore order of values (and index).
    ignore_index : bool
        Ignore index values and name.
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
        equal or a short explanation of why the series aren't equal, otherwise.

    Notes
    -----
    All values (including those of indices) must be copyable and ``__eq__`` must
    be such that a copy must equal its original. A value must equal itself
    unless it's ``NaN``. Values needn't be orderable or hashable (however
    pandas requires index values to be orderable and hashable). By consequence,
    this is not an efficient function, but it is flexible.
    '''
    result = _equals(series1, series2, ignore_order, ignore_index, all_close)
    if _return_reason:
        return result
    else:
        return result[0]