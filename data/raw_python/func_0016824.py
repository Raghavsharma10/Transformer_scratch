def nr_of_antenna(nbl, auto_correlations=False):
    """
    Compute the number of antenna for the
    given number of baselines. Can specify whether
    auto-correlations should be taken into
    account
    """
    t = 1 if auto_correlations is False else -1
    return int(t + math.sqrt(1 + 8*nbl)) // 2