def env_bool_flag(flag_name: str, strict: bool = False, env: t.Optional[t.Dict[str, str]] = None) -> bool:
    """
    Converts an environment variable into a boolean. Empty string (presence in env) is treated as ``True``.

    :param flag_name: an environment variable name
    :param strict: raise ``ValueError`` if a ``flag_name`` value connot be coerced into a boolean in obvious way
    :param env: a dict with environment variables, default is ``os.environ``
    :return: ``True`` if ``flag_name`` is thruthy, ``False`` otherwise.

    :raises ValueError: if ``strict`` specified and ``val`` got anything except ``['', 0, 1, true, false, True, False]``
    """
    env = env or os.environ
    sentinel = object()
    val = env.get(flag_name, sentinel)

    if val is sentinel:
        return False

    return coerce_str_to_bool(val, strict=strict)