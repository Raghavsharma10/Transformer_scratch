def is_json(string):
    """
    Check if a string is a valid json.

    :param string: String to check.
    :type string: str
    :return: True if json, false otherwise
    :rtype: bool
    """
    if not is_full_string(string):
        return False
    if bool(JSON_WRAPPER_RE.search(string)):
        try:
            return isinstance(json.loads(string), dict)
        except (TypeError, ValueError, OverflowError):
            return False
    return False