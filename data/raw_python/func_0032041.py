def _get_bin_width(stdev, count):
    """Return the histogram's optimal bin width based on Sturges

    http://www.jstor.org/pss/2965501
    """

    w = int(round((3.5 * stdev) / (count ** (1.0 / 3))))
    if w:
        return w
    else:
        return 1