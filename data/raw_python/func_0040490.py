def prepare_raise(func):
    """
    Just a short decorator which shrinks
    full ``raise (E, V, T)`` form into proper ``raise E(V), T``.
    """

    @functools.wraps(func)
    def decorator(type_, value=None, traceback=None):
        if value is not None and isinstance(type_, Exception):
            raise TypeError("instance exception may not have a separate value")

        if value is None:
            if isinstance(type_, Exception):
                error = type_
            else:
                error = type_()
        else:
            error = type_(value)
        func(error, value, traceback)

    return decorator