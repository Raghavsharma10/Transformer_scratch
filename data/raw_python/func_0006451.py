def logical_or(f1, f2):  # function factory
    '''Logical or from functions.

    Parameters
    ----------
    f1, f2 : function
        Function that takes array and returns true or false for each item in array.

    Returns
    -------
    Function.
    '''
    def f(value):
        return np.logical_or(f1(value), f2(value))
    f.__name__ = "(" + f1.__name__ + "_or_" + f2.__name__ + ")"
    return f