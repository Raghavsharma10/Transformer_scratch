def find_server(region='EU-London', mode=None):
    """
    Returns `(address, token)`, both strings.

    `mode` is the game mode of the requested server. It can be
    `'party'`, `'teams'`, `'experimental'`, or `None` for "Free for all".

    The returned `address` is in `'IP:port'` format.

    Makes a request to http://m.agar.io to get address and token.
    """
    if mode:
        region = '%s:%s' % (region, mode)
    opener = urllib.request.build_opener()
    opener.addheaders = default_headers
    data = '%s\n%s' % (region, handshake_version)
    return opener.open('http://m.agar.io/', data=data.encode()) \
        .read().decode().split('\n')[0:2]