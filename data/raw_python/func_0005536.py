def url_equals(a: str, b: str) -> bool:
    """
    Compares two URLs/paths and returns True if they point to same URI.
    For example, querystring parameters can be different order but URLs are still equal.
    :param a: URL/path
    :param b: URL/path
    :return: True if URLs/paths are equal
    """
    from urllib.parse import urlparse, parse_qsl
    a2 = list(urlparse(a))
    b2 = list(urlparse(b))
    a2[4] = dict(parse_qsl(a2[4]))
    b2[4] = dict(parse_qsl(b2[4]))
    return a2 == b2