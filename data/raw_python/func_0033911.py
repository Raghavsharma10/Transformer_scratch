def gcommer_donate(address, token, *_):
    """
    Donate a token for this server address.
    `address` and `token` should be the return values from find_server().
    """
    token = urllib.request.quote(token)
    url = 'http://at.gcommer.com/donate?server=%s&token=%s' % (address, token)
    response = urllib.request.urlopen(url).read().decode()
    return json.loads(response)['msg']