def is_valid_version_ip(param):
    """Checks if the parameter is a valid ip version value.

    :param param: Value to be validated.

    :return: True if the parameter has a valid ip version value, or False otherwise.
    """
    if param is None:
        return False

    if param == IP_VERSION.IPv4[0] or param == IP_VERSION.IPv6[0]:
        return True

    return False