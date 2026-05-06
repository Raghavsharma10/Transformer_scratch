def get_party_address(party_token):
    """
    Returns the address (`'IP:port'` string) of the party server.

    To generate a `party_token`:
    ```
    from agarnet.utils import find_server
    _, token = find_server(mode='party')
    ```

    Makes a request to http://m.agar.io/getToken to get the address.
    """
    opener = urllib.request.build_opener()
    opener.addheaders = default_headers
    try:
        data = party_token.encode()
        return opener.open('http://m.agar.io/getToken', data=data) \
            .read().decode().split('\n')[0]
    except urllib.error.HTTPError:
        raise ValueError('Invalid token "%s" (maybe timed out,'
                         ' try creating a new one)' % party_token)