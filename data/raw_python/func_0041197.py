def authenticate(previous_token = None):
    """ Authenticate the client to the server """

    # if we already have a session token, try to authenticate with it
    if previous_token != None:
        headers = server_connection.request("authenticate", {
            'session_token' : previous_token,
            'repository'    : config['repository']})[1] # Only care about headers

        if headers['status'] == 'ok':
            return previous_token

    # If the session token has expired, or if we don't have one, re-authenticate

    headers = server_connection.request("begin_auth", {'repository' : config['repository']})[1] # Only care about headers

    if headers['status'] == 'ok':
        signature = base64.b64encode(pysodium.crypto_sign_detached(headers['auth_token'].decode('utf-8'), config['private_key']))
        headers = server_connection.request("authenticate", {
            'auth_token' : headers['auth_token'],
            'signature'  : signature,
            'user'       : config['user'],
            'repository' : config['repository']})[1] # Only care about headers

        if headers['status'] == 'ok': return headers['session_token']
    raise SystemExit('Authentication failed')