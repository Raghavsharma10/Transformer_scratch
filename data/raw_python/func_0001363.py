def coerce_str_to_bool(val: t.Union[str, int, bool, None], strict: bool = False) -> bool:
    """
    Converts a given string ``val`` into a boolean.

    :param val: any string representation of boolean
    :param strict: raise ``ValueError`` if ``val`` does not look like a boolean-like object
    :return: ``True`` if ``val`` is thruthy, ``False`` otherwise.

    :raises ValueError: if ``strict`` specified and ``val`` got anything except
     ``['', 0, 1, true, false, on, off, True, False]``
    """
    if isinstance(val, str):
        val = val.lower()

    flag = ENV_STR_BOOL_COERCE_MAP.get(val, None)

    if flag is not None:
        return flag

    if strict:
        raise ValueError('Unsupported value for boolean flag: `%s`' % val)

    return bool(val)