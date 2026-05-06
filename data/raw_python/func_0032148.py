def safe_uriref(text):
    """ Escape a URL properly. """
    url_ = url.parse(text).sanitize().deuserinfo().canonical()
    return URIRef(url_.punycode().unicode())