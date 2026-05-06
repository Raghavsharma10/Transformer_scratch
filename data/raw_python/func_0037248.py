def gmdaOnes(shape, dtype, mask=None, numGhosts=1):
    """
    ghosted distributed array one constructor
    @param shape the shape of the array
    @param dtype the numpy data type
    @param numGhosts the number of ghosts (>= 0)
    """
    res = GhostedMaskedDistArray(shape, dtype)
    res.mask = mask
    res.setNumberOfGhosts(numGhosts)
    res[:] = 1
    return res