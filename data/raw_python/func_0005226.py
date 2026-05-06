def get_path(name, *default):
    # type: (str, Any) -> Any
    """ Get config value as path relative to the project directory.

    This allows easily defining the project configuration within the fabfile
    as always relative to that fabfile.

    Args:
        name (str):
            The name of the config value containing the path.
        *default (Any):
            If given and the key doesn't not exist, this will be returned
            instead. If it's not given and the config value does not exist,
            AttributeError will be raised

    Returns:
        The requested config value. This is one of the global values defined
        in this file. If the value does not exist it will return `default` if
        give or raise `AttributeError`.

    Raises:
        AttributeError: If the value does not exist and `default` was not given.
    """
    global g_config

    value = get(name, *default)

    if value is None:
        return None

    return proj_path(value)