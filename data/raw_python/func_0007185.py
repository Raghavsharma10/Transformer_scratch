def bool(name, execute_bool=True, default=None):
    """Only execute the function if the boolean variable is set.

    Args:
        name: The name of the environment variable
        execute_bool: The boolean value to execute the function on
        default: The default value if the environment variable is not set (respects `execute_bool`)

    Returns:
        The function return value or `None` if the function was skipped.
    """
    def wrapped(func):
        @functools.wraps(func)
        def _decorator(*args, **kwargs):
            if core.isset(name) and core.bool(name) == execute_bool:
                return func(*args, **kwargs)
            elif default is not None and default == execute_bool:
                return func(*args, **kwargs)
        return _decorator
    return wrapped