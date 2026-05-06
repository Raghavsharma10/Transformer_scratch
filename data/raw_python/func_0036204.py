def basemz(df):
    """
    The mz of the most abundant ion.
    """
    # returns the
    d = np.array(df.columns)[df.values.argmax(axis=1)]
    return Trace(d, df.index, name='basemz')