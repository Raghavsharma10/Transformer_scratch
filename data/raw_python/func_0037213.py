def mdaArray(arry, dtype=numpy.float, mask=None):
    """
    Array constructor for masked distributed array
    @param arry numpy-like array
    @param mask mask array (or None if all data elements are valid)
    """
    a = numpy.array(arry, dtype)
    res = MaskedDistArray(a.shape, a.dtype)
    res[:] = a
    res.mask = mask
    return res