def to_frame(data_list, exc_cols=None, **kwargs):
    """
    Dict in Python 3.6 keeps insertion order, but cannot be relied upon
    This method is to keep column names in order
    In Python 3.7 this method is redundant

    Args:
        data_list: list of dict
        exc_cols: exclude columns

    Returns:
        pd.DataFrame

    Example:
        >>> d_list = [
        ...     dict(sid=1, symbol='1 HK', price=89),
        ...     dict(sid=700, symbol='700 HK', price=350)
        ... ]
        >>> to_frame(d_list)
           sid  symbol  price
        0    1    1 HK     89
        1  700  700 HK    350
        >>> to_frame(d_list, exc_cols=['price'])
           sid  symbol
        0    1    1 HK
        1  700  700 HK
    """
    from collections import OrderedDict

    return pd.DataFrame(
        pd.Series(data_list).apply(OrderedDict).tolist(), **kwargs
    ).drop(columns=[] if exc_cols is None else exc_cols)