def daZeros(shap, dtype=numpy.float):
    """
    Zero constructor for numpy distributed array
    @param shap the shape of the array
    @param dtype the numpy data type
    """
    res = DistArray(shap, dtype)
    res[:] = 0
    return res