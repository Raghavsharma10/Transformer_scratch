def get_mirrors(hostname=None):
    """Return the list of mirrors from the last record found on the DNS
    entry::

    >>> from pip.index import get_mirrors
    >>> get_mirrors()
    ['a.pypi.python.org', 'b.pypi.python.org', 'c.pypi.python.org',
    'd.pypi.python.org']

    Originally written for the distutils2 project by Alexis Metaireau.
    """
    if hostname is None:
        hostname = DEFAULT_MIRROR_URL

    # return the last mirror registered on PyPI.
    try:
        hostname = socket.gethostbyname_ex(hostname)[0]
    except socket.gaierror:
        return []
    end_letter = hostname.split(".", 1)

    # determine the list from the last one.
    return ["%s.%s" % (s, end_letter[1]) for s in string_range(end_letter[0])]