def uniq2orderipix(uniq):
    """
    convert a HEALPix pixel coded as a NUNIQ number
    to a (norder, ipix) tuple
    """
    order = ((np.log2(uniq//4)) // 2)
    order = order.astype(int)
    ipix = uniq - 4 * (4**order)

    return order, ipix