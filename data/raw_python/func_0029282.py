def is_slug(string, sign='-'):
    """
    Checks if a given string is a slug.

    :param string: String to check.
    :type string: str
    :param sign: Join sign used by the slug.
    :type sign: str
    :return: True if slug, false otherwise.
    """
    if not is_full_string(string):
        return False
    rex = r'^([a-z\d]+' + re.escape(sign) + r'?)*[a-z\d]$'
    return re.match(rex, string) is not None