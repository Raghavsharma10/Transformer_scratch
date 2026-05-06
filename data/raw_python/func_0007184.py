def isset(name):
    """Only execute the function if the variable is set.

    Args:
        name: The name of the environment variable

    Returns:
        The function return value or `None` if the function was skipped.
    """
    def wrapped(func):
        @functools.wraps(func)
        def _decorator(*args, **kwargs):
            if core.isset(name):
                return func(*args, **kwargs)
        return _decorator
    return wrapped