def gmdaZeros(shape, dtype, mask=None, numGhosts=1):
    """
    ghosted distributed array zero constructor
    @param shape the shape of the array
    @param dtype the numpy data type
    @param numGhosts the number of ghosts (>= 0)
    """
    res = GhostedMaskedDistArray(shape, dtype)
    res.mas = mask
    res.setNumberOfGhosts(numGhosts)
    res[:] = 0
    return res