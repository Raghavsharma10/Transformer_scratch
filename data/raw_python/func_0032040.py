def get_histogram_bins(min_, max_, std, count):
    """
    Return optimal bins given the input parameters

    """

    width = _get_bin_width(std, count)
    count = int(round((max_ - min_) / width) + 1)

    if count:
        bins = [i * width + min_ for i in xrange(1, count + 1)]
    else:
        bins = [min_]

    return bins