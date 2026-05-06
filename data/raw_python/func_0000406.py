def deprecated(func, msg=None):
    """
    A decorator which can be used to mark functions
    as deprecated.It will result in a deprecation warning being shown
    when the function is used.
    """

    message = msg or "Use of deprecated function '{}`.".format(func.__name__)

    @functools.wraps(func)
    def wrapper_func(*args, **kwargs):
        warnings.warn(message, DeprecationWarning, stacklevel=2)
        return func(*args, **kwargs)

    return wrapper_func