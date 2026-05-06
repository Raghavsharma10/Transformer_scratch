def transport_from_url(url):
    """ Create a transport for the given URL.
    """
    if '/' not in url and ':' in url and url.rsplit(':')[-1].isdigit():
        url = 'scgi://' + url
    url = urlparse.urlsplit(url, scheme="scgi", allow_fragments=False)  # pylint: disable=redundant-keyword-arg

    try:
        transport = TRANSPORTS[url.scheme.lower()]
    except KeyError:
        if not any((url.netloc, url.query)) and url.path.isdigit():
            # Support simplified "domain:port" URLs
            return transport_from_url("scgi://%s:%s" % (url.scheme, url.path))
        else:
            raise URLError("Unsupported scheme in URL %r" % url.geturl())
    else:
        return transport(url)