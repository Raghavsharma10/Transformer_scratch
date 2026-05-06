def daOnes(shap, dtype=numpy.float):
    """
    One constructor for numpy distributed array
    @param shap the shape of the array
    @param dtype the numpy data type
    """
    res = DistArray(shap, dtype)
    res[:] = 1
    return res