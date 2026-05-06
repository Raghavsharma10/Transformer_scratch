def within_set(df, items=None):
    """
    Assert that df is a subset of items

    Parameters
    ==========

    df : DataFrame
    items : dict
        mapping of columns (k) to array-like of values (v) that
        ``df[k]`` is expected to be a subset of
    """
    for k, v in items.items():
        if not df[k].isin(v).all():
            raise AssertionError
    return df