def port(alias_name, default=None, allow_none=False):
    """Get the port from the docker link alias or return the default.

    Args:
        alias_name: The docker link alias
        default: The default value if the link isn't available
        allow_none: If the return value can be `None` (i.e. optional)

    Examples:
        Assuming a Docker link was created with ``docker --link postgres:db``
        and the resulting environment variable is ``DB_PORT=tcp://172.17.0.82:5432``.

        >>> envitro.docker.port('DB')
        5432
    """
    warnings.warn('Will be removed in v1.0', DeprecationWarning, stacklevel=2)
    try:
        return int(_split_docker_link(alias_name)[2])
    except KeyError as err:
        if default or allow_none:
            return default
        else:
            raise err