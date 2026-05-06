def is_snake_case(string, separator='_'):
    """
    Checks if a string is formatted as snake case.
    A string is considered snake case when:

    * it's composed only by lowercase letters ([a-z]), underscores (or provided separator) \
    and optionally numbers ([0-9])
    * it does not start/end with an underscore (or provided separator)
    * it does not start with a number


    :param string: String to test.
    :type string: str
    :param separator: String to use as separator.
    :type separator: str
    :return: True for a snake case string, false otherwise.
    :rtype: bool
    """
    if is_full_string(string):
        re_map = {
            '_': SNAKE_CASE_TEST_RE,
            '-': SNAKE_CASE_TEST_DASH_RE
        }
        re_template = '^[a-z]+([a-z\d]+{sign}|{sign}[a-z\d]+)+[a-z\d]+$'
        r = re_map.get(separator, re.compile(re_template.format(sign=re.escape(separator))))
        return bool(r.search(string))
    return False