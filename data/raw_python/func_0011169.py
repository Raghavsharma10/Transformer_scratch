def convert_type(ty, default=None):
    """Converts a callable or python ty into the most appropriate param
    ty.
    """
    if isinstance(ty, tuple):
        return Tuple(ty)
    if isinstance(ty, ParamType):
        return ty
    guessed_type = False
    if ty is None and default is not None:
        ty = type(default)
        guessed_type = True
    if ty is text_type or ty is str or ty is None:
        return STRING
    if ty is int:
        return INT
    # Booleans are only okay if not guessed.  This is done because for
    # flags the default value is actually a bit of a lie in that it
    # indicates which of the flags is the one we want.  See get_default()
    # for more information.
    if ty is bool and not guessed_type:
        return BOOL
    if ty is float:
        return FLOAT
    if guessed_type:
        return STRING

    # Catch a common mistake
    if __debug__:
        try:
            if issubclass(ty, ParamType):
                raise AssertionError('Attempted to use an uninstantiated '
                                     'parameter type (%s).' % ty)
        except TypeError:
            pass
    return FuncParamType(ty)