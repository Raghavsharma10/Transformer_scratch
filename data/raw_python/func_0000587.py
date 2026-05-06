def convert_python_regex_to_ecma(value, flags=[]):
    """Convert Python regex to ECMA 262 regex.

    If given value is already ECMA regex it will be returned unchanged.

    :param string value: Python regex.
    :param list flags: List of flags (allowed flags: `re.I`, `re.M`)
    :return: ECMA 262 regex
    :rtype: str

    """
    if is_ecma_regex(value):
        return value

    result_flags = [PYTHON_TO_ECMA_FLAGS[f] for f in flags]
    result_flags = ''.join(result_flags)

    return '/{value}/{flags}'.format(value=value, flags=result_flags)