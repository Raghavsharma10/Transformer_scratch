def parse_url(url, encoding=None):
    """Return urlparsed url from the given argument (which could be an already
    parsed url)
    """
    return url if isinstance(url, urlparse.ParseResult) else \
        urlparse.urlparse(unicode_to_str(url, encoding))