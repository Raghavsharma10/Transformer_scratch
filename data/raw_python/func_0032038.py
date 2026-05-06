def percentile(data, n):
    """Return the n-th percentile of the given data

    Assume that the data are already sorted

    """

    size = len(data)
    idx = (n / 100.0) * size - 0.5

    if idx < 0 or idx > size:
        raise StatisticsError("Too few data points ({}) for {}th percentile".format(size, n))

    return data[int(idx)]