def get_histogram(data):
    """Return the histogram relative to the given data

    Assume that the data are already sorted

    """

    count = len(data)

    if count < 2:
        raise StatisticsError('Too few data points ({}) for get_histogram'.format(count))

    min_ = data[0]
    max_ = data[-1]
    std = stdev(data)

    bins = get_histogram_bins(min_, max_, std, count)

    res = {x: 0 for x in bins}

    for value in data:
        for bin_ in bins:
            if value <= bin_:
                res[bin_] += 1
                break

    return sorted(iteritems(res))