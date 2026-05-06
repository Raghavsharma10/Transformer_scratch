def fetch_and_parse(method, uri, params_prefix=None, **params):
    """Fetch the given uri and return the root Element of the response."""
    doc = ElementTree.parse(fetch(method, uri, params_prefix, **params))
    return _parse(doc.getroot())