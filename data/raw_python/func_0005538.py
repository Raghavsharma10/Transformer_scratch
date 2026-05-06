def url_host(url: str) -> str:
    """
    Parses hostname from URL.
    :param url: URL
    :return: hostname
    """
    from urllib.parse import urlparse
    res = urlparse(url)
    return res.netloc.split(':')[0] if res.netloc else ''