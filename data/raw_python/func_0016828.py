def dict_array_bytes(ary, template):
    """
    Return the number of bytes required by an array

    Arguments
    ---------------
    ary : dict
        Dictionary representation of an array
    template : dict
        A dictionary of key-values, used to replace any
        string values in the array with concrete integral
        values

    Returns
    -----------
    The number of bytes required to represent
    the array.
    """
    shape = shape_from_str_tuple(ary['shape'], template)
    dtype = dtype_from_str(ary['dtype'], template)

    return array_bytes(shape, dtype)