def logical_not(f):  # function factory
    '''Logical not from functions.

    Parameters
    ----------
    f1, f2 : function
        Function that takes array and returns true or false for each item in array.

    Returns
    -------
    Function.
    '''
    def f(value):
        return np.logical_not(f(value))
    f.__name__ = "not_" + f.__name__
    return f