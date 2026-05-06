def merge(tup):
    """Merge several timeseries
    Arguments:
      tup: sequence of Timeseries, with the same shape except for axis 0
    Returns: 
      Resulting merged timeseries which can have duplicate time points.
    """
    if not all(tuple(ts.shape[1:] == tup[0].shape[1:] for ts in tup[1:])):
        raise ValueError('Timeseries to merge must have compatible shapes')
    indices = np.vstack(tuple(ts.tspan for ts in tup)).argsort()
    return np.vstack((tup))[indices]