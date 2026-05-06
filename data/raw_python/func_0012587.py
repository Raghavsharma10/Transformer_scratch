def dict_2_mat(data, fill = True):
    """
    Creates a NumPy array from a dictionary with only integers as keys and
    NumPy arrays as values. Dimension 0 of the resulting array is formed from
    data.keys(). Missing values in keys can be filled up with np.nan (default)
    or ignored.

    Parameters
    ----------
    data : dict
        a dictionary with integers as keys and array-likes of the same shape
        as values
    fill : boolean
        flag specifying if the resulting matrix will keep a correspondence
        between dictionary keys and matrix indices by filling up missing keys
        with matrices of NaNs. Defaults to True

    Returns
    -------
    numpy array with one more dimension than the values of the input dict
    """
    if any([type(k) != int for k in list(data.keys())]):
        raise RuntimeError("Dictionary cannot be converted to matrix, " +
                            "not all keys are ints")
    base_shape = np.array(list(data.values())[0]).shape
    result_shape = list(base_shape)
    if fill:
        result_shape.insert(0, max(data.keys()) + 1)
    else:
        result_shape.insert(0, len(list(data.keys())))
    result = np.empty(result_shape) + np.nan
        
    for (i, (k, v)) in enumerate(data.items()):
        v = np.array(v)
        if v.shape != base_shape:
            raise RuntimeError("Dictionary cannot be converted to matrix, " +
                                        "not all values have same dimensions")
        result[fill and [k][0] or [i][0]] = v
    return result