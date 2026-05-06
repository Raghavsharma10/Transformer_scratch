def geturl(urllib2_resp):
    """
    Use instead of urllib.addinfourl.geturl(), which appears to have
    some issues with dropping the double slash for certain schemes
    (e.g. file://).  This implementation is probably over-eager, as it
    always restores '://' if it is missing, and it appears some url
    schemata aren't always followed by '//' after the colon, but as
    far as I know pip doesn't need any of those.
    The URI RFC can be found at: http://tools.ietf.org/html/rfc1630

    This function assumes that
        scheme:/foo/bar
    is the same as
        scheme:///foo/bar
    """
    url = urllib2_resp.geturl()
    scheme, rest = url.split(':', 1)
    if rest.startswith('//'):
        return url
    else:
        # FIXME: write a good test to cover it
        return '%s://%s' % (scheme, rest)