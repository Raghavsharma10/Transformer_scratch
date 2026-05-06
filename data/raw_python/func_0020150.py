def parse_backend(backend):
    """Converts the "backend" into the database connection parameters.
It returns a (scheme, host, params) tuple."""
    r = urlparse.urlsplit(backend)
    scheme, host = r.scheme, r.netloc
    path, query = r.path, r.query
    if path and not query:
        query, path = path, ''
        if query:
            if query.find('?'):
                path = query
            else:
                query = query[1:]
    if query:
        params = dict(urlparse.parse_qsl(query))
    else:
        params = {}

    return scheme, host, params