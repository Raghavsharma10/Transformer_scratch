def parse_url_to_dict(url):
    """Parse a url and return a dict with keys for all of the parts.

    The urlparse function() returns a wacky combination of a namedtuple
    with properties.

    """
    p = urlparse(url)

    return {
        'scheme': p.scheme,
        'netloc': p.netloc,
        'path': p.path,
        'params': p.params,
        'query': p.query,
        'fragment': p.fragment,
        'username': p.username,
        'password': p.password,
        'hostname': p.hostname,
        'port': p.port
    }