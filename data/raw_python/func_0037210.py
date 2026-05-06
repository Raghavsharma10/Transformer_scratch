def daArray(arry, dtype=numpy.float):
    """
    Array constructor for numpy distributed array
    @param arry numpy-like array
    """
    a = numpy.array(arry, dtype)
    res = DistArray(a.shape, a.dtype)
    res[:] = a
    return res