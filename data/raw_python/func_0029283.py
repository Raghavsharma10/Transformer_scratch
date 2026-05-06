def camel_case_to_snake(string, separator='_'):
    """
    Convert a camel case string into a snake case one.
    (The original string is returned if is not a valid camel case string)

    :param string: String to convert.
    :type string: str
    :param separator: Sign to use as separator.
    :type separator: str
    :return: Converted string.
    :rtype: str
    """
    if not is_string(string):
        raise TypeError('Expected string')
    if not is_camel_case(string):
        return string
    return CAMEL_CASE_REPLACE_RE.sub(lambda m: m.group(1) + separator, string).lower()