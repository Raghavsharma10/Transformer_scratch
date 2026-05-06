def mean(data):
    """Return the sample arithmetic mean of data."""
    #: http://stackoverflow.com/a/27758326
    n = len(data)
    if n < 1:
        raise ValueError('mean requires at least one data point')
    return sum(data)/n