def make_url(url, *paths):
    """Joins individual URL strings together, and returns a single string.
    """
    for path in paths:
        url = re.sub(r'/?$', re.sub(r'^/?', '/', path), url)
    return url