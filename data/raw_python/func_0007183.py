def write(name, value):
    """Temporarily change or set the environment variable during the execution of a function.

    Args:
        name: The name of the environment variable
        value: A value to set for the environment variable

    Returns:
        The function return value.
    """
    def wrapped(func):
        @functools.wraps(func)
        def _decorator(*args, **kwargs):
            existing_env = core.read(name, allow_none=True)
            core.write(name, value)
            func_val = func(*args, **kwargs)
            core.write(name, existing_env)
            return func_val
        return _decorator
    return wrapped