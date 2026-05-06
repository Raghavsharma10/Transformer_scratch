def logical_and(f1, f2):  # function factory
    '''Logical and from functions.

    Parameters
    ----------
    f1, f2 : function
        Function that takes array and returns true or false for each item in array.

    Returns
    -------
    Function.

    Usage:
    filter_func=logical_and(is_data_record, is_data_from_channel(4))  # new filter function
    filter_func(array) # array that has Data Records from channel 4
    '''
    def f(value):
        return np.logical_and(f1(value), f2(value))
    f.__name__ = "(" + f1.__name__ + "_and_" + f2.__name__ + ")"
    return f