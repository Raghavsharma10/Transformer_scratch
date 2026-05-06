def get(name, *default):
    # type: (str, Any) -> Any
    """ Get config value with the given name and optional default.

    Args:
        name (str):
            The name of the config value.
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

    curr = g_config
    for part in name.split('.'):
        if part in curr:
            curr = curr[part]
        elif default:
            return default[0]
        else:
            raise AttributeError("Config value '{}' does not exist".format(
                name
            ))

    return curr