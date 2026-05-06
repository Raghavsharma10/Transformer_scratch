def is_method(arg, min_arity=None, max_arity=None):
    """Check if argument is a method.

    Optionally, we can also check if minimum or maximum arities
    (number of accepted arguments) match given minimum and/or maximum.
    """
    if not callable(arg):
        return False

    if not any(is_(arg) for is_ in (inspect.ismethod,
                                    inspect.ismethoddescriptor,
                                    inspect.isbuiltin)):
        return False

    try:
        argnames, varargs, kwargs, defaults = getargspec(arg)
    except TypeError:
        # On CPython 2.x, built-in methods of file aren't inspectable,
        # so if it's file.read() or file.write(), we can't tell it for sure.
        # Given how this check is being used, assuming the best is probably
        # all we can do here.
        return True
    else:
        if argnames and argnames[0] == 'self':
            argnames = argnames[1:]

    if min_arity is not None:
        actual_min_arity = len(argnames) - len(defaults or ())
        assert actual_min_arity >= 0, (
            "Minimum arity of %r found to be negative (got %s)!" % (
                arg, actual_min_arity))
        if int(min_arity) != actual_min_arity:
            return False

    if max_arity is not None:
        actual_max_arity = sys.maxsize if varargs or kwargs else len(argnames)
        if int(max_arity) != actual_max_arity:
            return False

    return True