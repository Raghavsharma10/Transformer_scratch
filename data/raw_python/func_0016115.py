def assert_valid_id(id):
    """Checks if an id is the correct format that Marathon expects. Raises ValueError if not valid.

    :param str id: App or group id.

    :rtype: str
    """
    if id is None:
        return
    if not ID_PATTERN.match(id.strip('/')):
        raise ValueError(
            'invalid id (allowed: lowercase letters, digits, hyphen, ".", ".."): %r' % id)
    return id