def geometric_mean(data):
    """Return the geometric mean of data
    """

    if not data:
        raise StatisticsError('geometric_mean requires at least one data point')

    # in order to support negative or null values
    data = [x if x > 0 else math.e if x == 0 else 1.0 for x in data]

    return math.pow(math.fabs(functools.reduce(operator.mul, data)), 1.0 / len(data))