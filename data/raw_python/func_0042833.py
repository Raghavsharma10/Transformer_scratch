def init(scope):
    """
    Copy all values of scope into the class SinonGlobals
    Args:
        scope (eg. locals() or globals())
    Return:
        SinonGlobals instance
    """
    class SinonGlobals(object): #pylint: disable=too-few-public-methods
        """
        A fully empty class
        External can push the whole `scope` into this class through global function init()
        """
        pass

    global CPSCOPE #pylint: disable=global-statement
    CPSCOPE = SinonGlobals()
    funcs = [obj for obj in scope.values() if isinstance(obj, FunctionType)]
    for func in funcs:
        setattr(CPSCOPE, func.__name__, func)
    return CPSCOPE