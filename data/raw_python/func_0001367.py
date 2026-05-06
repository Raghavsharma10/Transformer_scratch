def is_production(flag_name: str = 'PRODUCTION', strict: bool = False):
    """
    Reads env ``PRODUCTION`` variable as a boolean.

    :param flag_name: environment variable name
    :param strict: raise a ``ValueError`` if variable does not look like a normal boolean
    :return: ``True`` if has truthy ``PRODUCTION`` env, ``False`` otherwise
    """
    return env_bool_flag(flag_name, strict=strict)