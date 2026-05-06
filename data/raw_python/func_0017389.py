def ts_merge(series):
    '''Merge timeseries into a new :class:`~.TimeSeries` instance.

    :parameter series: an iterable over :class:`~.TimeSeries`.
    '''
    series = iter(series)
    ts = next(series)
    return ts.merge(series)