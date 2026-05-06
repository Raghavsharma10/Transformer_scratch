def within_range(df, items=None):
    """
    Assert that a DataFrame is within a range.

    Parameters
    ==========
    df : DataFame
    items : dict
        mapping of columns (k) to a (low, high) tuple (v)
        that ``df[k]`` is expected to be between.
    """
    for k, (lower, upper) in items.items():
        if (lower > df[k]).any() or (upper < df[k]).any():
            raise AssertionError
    return df