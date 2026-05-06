def gdaArray(arry, dtype, numGhosts=1):
    """
    ghosted distributed array constructor
    @param arry numpy-like array
    @param numGhosts the number of ghosts (>= 0)
    """
    a = numpy.array(arry, dtype)
    res = GhostedDistArray(a.shape, a.dtype)
    res.setNumberOfGhosts(numGhosts)
    res[:] = a
    return res