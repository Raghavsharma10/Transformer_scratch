def is_dockerized(flag_name: str = 'DOCKERIZED', strict: bool = False):
    """
    Reads env ``DOCKERIZED`` variable as a boolean.

    :param flag_name: environment variable name
    :param strict: raise a ``ValueError`` if variable does not look like a normal boolean
    :return: ``True`` if has truthy ``DOCKERIZED`` env, ``False`` otherwise
    """
    return env_bool_flag(flag_name, strict=strict)