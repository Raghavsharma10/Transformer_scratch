def gdaOnes(shape, dtype, numGhosts=1):
    """
    ghosted distributed array one constructor
    @param shape the shape of the array
    @param dtype the numpy data type
    @param numGhosts the number of ghosts (>= 0)
    """
    res = GhostedDistArray(shape, dtype)
    res.setNumberOfGhosts(numGhosts)
    res[:] = 1
    return res