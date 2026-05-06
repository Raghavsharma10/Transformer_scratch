def logical_xor(f1, f2):  # function factory
    '''Logical xor from functions.

    Parameters
    ----------
    f1, f2 : function
        Function that takes array and returns true or false for each item in array.

    Returns
    -------
    Function.
    '''
    def f(value):
        return np.logical_xor(f1(value), f2(value))
    f.__name__ = "(" + f1.__name__ + "_xor_" + f2.__name__ + ")"
    return f