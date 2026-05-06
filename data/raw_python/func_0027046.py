def getargspec(func):
    """Variation of inspect.getargspec that works for more functions.

    This function works for Cythonized, non-cpdef functions, which expose argspec information but
    are not accepted by getargspec. It also works for Python 3 functions that use annotations, which
    are simply ignored. However, keyword-only arguments are not supported.

    """
    if inspect.ismethod(func):
        func = func.__func__
    # Cythonized functions have a .__code__, but don't pass inspect.isfunction()
    try:
        code = func.__code__
    except AttributeError:
        raise TypeError("{!r} is not a Python function".format(func))
    if hasattr(code, "co_kwonlyargcount") and code.co_kwonlyargcount > 0:
        raise ValueError("keyword-only arguments are not supported by getargspec()")
    args, varargs, varkw = inspect.getargs(code)
    return inspect.ArgSpec(args, varargs, varkw, func.__defaults__)