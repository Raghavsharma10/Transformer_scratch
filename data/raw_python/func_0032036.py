def harmonic_mean(data):
    """Return the harmonic mean of data
    """

    if not data:
        raise StatisticsError('harmonic_mean requires at least one data point')

    divisor = sum(map(lambda x: 1.0 / x if x else 0.0, data))
    return len(data) / divisor if divisor else 0.0