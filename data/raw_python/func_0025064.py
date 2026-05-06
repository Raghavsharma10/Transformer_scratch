def string_is_url(test_str):
    """ Test to see if a string is a URL or not, defined in this case as a string for which
    urlparse returns a scheme component

    >>> string_is_url('somestring')
    False
    >>> string_is_url('https://some.domain.org/path')
    True
    """
    parsed = urlparse.urlparse(test_str)
    return parsed.scheme is not None and parsed.scheme != ''