def dict_array_bytes_required(arrays, template):
    """
    Return the number of bytes required by
    a dictionary of arrays.

    Arguments
    ---------------
    arrays : list
        A list of dictionaries defining the arrays
    template : dict
        A dictionary of key-values, used to replace any
        string values in the arrays with concrete integral
        values

    Returns
    -----------
    The number of bytes required to represent
    all the arrays.
    """
    return np.sum([dict_array_bytes(ary, template)
        for ary in arrays])