def ts_func(f):
    """
    This wraps a function that would normally only accept an array
    and allows it to operate on a DataFrame. Useful for applying
    numpy functions to DataFrames.
    """
    def wrap_func(df, *args):
        # TODO: should vectorize to apply over all columns?
        return Chromatogram(f(df.values, *args), df.index, df.columns)
    return wrap_func