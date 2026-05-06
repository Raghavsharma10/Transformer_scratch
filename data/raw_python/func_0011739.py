def __clone_function(f, name=None):
    """Make a new version of a function that has its own independent copy 
    of any globals that it uses directly, and has its own name. 
    All other attributes are assigned from the original function.

    Args:
      f: the function to clone
      name (str):  the name for the new function (if None, keep the same name)

    Returns:
      A copy of the function f, having its own copy of any globals used

    Raises:
      SimValueError
    """
    if not isinstance(f, types.FunctionType):
        raise SimTypeError('Given parameter is not a function.')
    if name is None:
        name = f.__name__
    newglobals = f.__globals__.copy()
    globals_used = [x for x in f.__globals__ if x in f.__code__.co_names]
    for x in globals_used:
        gv = f.__globals__[x]
        if isinstance(gv, types.FunctionType):
            # Recursively clone any global functions used by this function.
            newglobals[x] = __clone_function(gv)
        elif isinstance(gv, types.ModuleType):
            newglobals[x] = gv
        else:
            # If it is something else, deep copy it.
            newglobals[x] = copy.deepcopy(gv)
    newfunc = types.FunctionType(
        f.__code__, newglobals, name, f.__defaults__, f.__closure__)
    return newfunc