def getCountKwargs(func):
    """ Returns a list ["count kwarg", "count_max kwarg"] for a
    given function. Valid combinations are defined in 
    `progress.validCountKwargs`.
    
    Returns None if no keyword arguments are found.
    """
    # Get all arguments of the function
    if hasattr(func, "__code__"):
        func_args = func.__code__.co_varnames[:func.__code__.co_argcount]
        for pair in validCountKwargs:
            if ( pair[0] in func_args and pair[1] in func_args ):
                return pair
    # else
    return None