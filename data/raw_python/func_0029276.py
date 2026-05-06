def is_url(string, allowed_schemes=None):
    """
    Check if a string is a valid url.

    :param string: String to check.
    :param allowed_schemes: List of valid schemes ('http', 'https', 'ftp'...). Default to None (any scheme is valid).
    :return: True if url, false otherwise
    :rtype: bool
    """
    if not is_full_string(string):
        return False
    valid = bool(URL_RE.search(string))
    if allowed_schemes:
        return valid and any([string.startswith(s) for s in allowed_schemes])
    return valid