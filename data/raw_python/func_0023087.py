def uniq2orderipix_lut(uniq):
    """
    ~30% faster than the method below
    Parameters
    ----------
    uniq

    Returns
    -------

    """
    order = log2_lut(uniq >> 2) >> 1
    ipix = uniq - (1 << (2 * (order + 1)))
    return order, ipix