def mdaZeros(shap, dtype=numpy.float, mask=None):
    """
    Zero constructor for masked distributed array
    @param shap the shape of the array
    @param dtype the numpy data type
    @param mask mask array (or None if all data elements are valid)
    """
    res = MaskedDistArray(shap, dtype)
    res[:] = 0
    res.mask = mask
    return res