def _ss(data, c=None):
    """Return sum of square deviations of sequence data.

    If ``c`` is None, the mean is calculated in one pass, and the deviations
    from the mean are calculated in a second pass. Otherwise, deviations are
    calculated from ``c`` as given. Use the second case with care, as it can
    lead to garbage results.
    """
    if c is None:
        c = mean(data)
    #print(data)
    ss = _sum((x-c)**2 for x in data)
    # The following sum should mathematically equal zero, but due to rounding
    # error may not.
    ss -= _sum((x-c) for x in data)**2/len(data)
    assert not ss < 0, 'negative sum of square deviations: %f' % ss
    return ss