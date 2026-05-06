def gmdaArray(arry, dtype, mask=None, numGhosts=1):
    """
    ghosted distributed array constructor
    @param arry numpy-like array
    @param numGhosts the number of ghosts (>= 0)
    """
    a = numpy.array(arry, dtype)
    res = GhostedMaskedDistArray(a.shape, a.dtype)
    res.mask = mask
    res.setNumberOfGhosts(numGhosts)
    res[:] = a
    return res