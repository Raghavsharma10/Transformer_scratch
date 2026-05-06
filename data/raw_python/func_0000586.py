def convert_ecma_regex_to_python(value):
    """Convert ECMA 262 regex to Python tuple with regex and flags.

    If given value is already Python regex it will be returned unchanged.

    :param string value: ECMA regex.
    :return: 2-tuple with `regex` and `flags`
    :rtype: namedtuple

    """
    if not is_ecma_regex(value):
        return PythonRegex(value, [])

    parts = value.split('/')
    flags = parts.pop()

    try:
        result_flags = [ECMA_TO_PYTHON_FLAGS[f] for f in flags]
    except KeyError:
        raise ValueError('Wrong flags "{}".'.format(flags))

    return PythonRegex('/'.join(parts[1:]), result_flags)