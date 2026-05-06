def build_digest_challenge(timestamp, secret, realm, opaque, stale):
    '''
    Builds a Digest challenge that may be sent as the value of the 'WWW-Authenticate' header
    in a 401 or 403 response.

    'opaque' may be any value - it will be returned by the client.

    'timestamp' will be incorporated and signed in the nonce - it may be retrieved from the
    client's authentication request using get_nonce_timestamp()
    '''
    nonce = calculate_nonce(timestamp, secret)

    return 'Digest %s' % format_parts(realm=realm, qop='auth', nonce=nonce,
                                      opaque=opaque, algorithm='MD5',
                                      stale=stale and 'true' or 'false')