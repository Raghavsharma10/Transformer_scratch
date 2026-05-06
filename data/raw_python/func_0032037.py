def kurtosis(data):
    """Return the kurtosis of the data's distribution

    """

    if not data:
        raise StatisticsError('kurtosis requires at least one data point')

    size = len(data)
    sd = stdev(data) ** 4

    if not sd:
        return 0.0

    mn = mean(data)
    return sum(map(lambda x: ((x - mn) ** 4 / sd), data)) / size - 3