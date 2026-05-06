def write(name, value):
    """Write a raw env value.

    A ``None`` value clears the environment variable.

    Args:
        name: The environment variable name
        value: The value to write
    """
    if value is not None:
        environ[name] = builtins.str(value)
    elif environ.get(name):
        del environ[name]