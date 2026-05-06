def add_url_suffix(url):
    """Add .com suffix to URL if none found."""
    url = url.rstrip('/')
    if not has_suffix(url):
        return '{0}.com'.format(url)
    return url