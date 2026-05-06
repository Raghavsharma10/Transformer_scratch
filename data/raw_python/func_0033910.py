def gcommer_claim(address=None):
    """
    Try to get a token for this server address.
    `address` has to be ip:port, e.g. `'1.2.3.4:1234'`
    Returns tuple(address, token)
    """
    if not address:
        # get token for any world
        # this is only useful for testing,
        # because that is exactly what m.agar.io does
        url = 'http://at.gcommer.com/status'
        text = urllib.request.urlopen(url).read().decode()
        j = json.loads(text)
        for address, num in j['status'].items():
            if num > 0:
                break  # address is now one of the listed servers with tokens
    url = 'http://at.gcommer.com/claim?server=%s' % address
    text = urllib.request.urlopen(url).read().decode()
    j = json.loads(text)
    token = j['token']
    return address, token