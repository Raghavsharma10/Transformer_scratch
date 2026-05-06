def _func_copy(f, newcode) :
    '''
    Return a copy of function f with a different __code__
    Because I can't find proper documentation on the
    correct signature of the types.FunctionType() constructor,
    I pass the minimum arguments then set the important
    dunder-values by direct assignment.

    Note you cannot assign __closure__, it is a "read-only attribute".
    Ergo, you should not apply _make_constants() to a function that
    has a closure!
    '''
    newf = types.FunctionType( newcode, f.__globals__ )
    newf.__annotations__ = f.__annotations__
    # newf.__closure__ = f.__closure__
    newf.__defaults__ = f.__defaults__
    newf.__doc__ = f.__doc__
    newf.__name__ = f.__name__
    newf.__kwdefaults__ = f.__kwdefaults__
    newf.__qualname__ = f.__qualname__
    return newf