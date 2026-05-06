def molmz(df, noise=10000):
    """
    The mz of the molecular ion.
    """
    d = ((df.values > noise) * df.columns).max(axis=1)
    return Trace(d, df.index, name='molmz')