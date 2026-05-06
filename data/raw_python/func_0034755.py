def build_authorization_request(username, method, uri, nonce_count, digest_challenge=None,
                                realm=None, nonce=None, opaque=None, password=None,
                                request_digest=None, client_nonce=None):
    '''
    Builds an authorization request that may be sent as the value of the 'Authorization'
    header in an HTTP request.

    Either a digest_challenge object (as returned from parse_digest_challenge) or its required
    component parameters (nonce, realm, opaque) must be provided.

    The nonce_count should be the last used nonce_count plus one.

    Either the password or the request_digest should be provided - if provided, the password
    will be used to generate a request digest. The client_nonce is optional - if not provided,
    a random value will be generated.
    '''
    if not client_nonce:
        client_nonce =  ''.join([random.choice('0123456789ABCDEF') for x in range(32)])

    if digest_challenge and (realm or nonce or opaque):
        raise Exception("Both digest_challenge and one or more of realm, nonce, and opaque"
                        "were sent.")

    if digest_challenge:
        if isinstance(digest_challenge, types.StringType):
            digest_challenge_header = digest_challenge
            digest_challenge = parse_digest_challenge(digest_challenge_header)
            if not digest_challenge:
                raise Exception("The provided digest challenge header could not be parsed: %s" %
                                digest_challenge_header)
        realm = digest_challenge.realm
        nonce = digest_challenge.nonce
        opaque = digest_challenge.opaque
    elif not (realm and nonce and opaque):
        raise Exception("Either digest_challenge or realm, nonce, and opaque must be sent.")

    if password and request_digest:
        raise Exception("Both password and calculated request_digest were sent.")
    elif not request_digest:
        if not password:
            raise Exception("Either password or calculated request_digest must be provided.")

        partial_digest = calculate_partial_digest(username, realm, password)
        request_digest = calculate_request_digest(method, partial_digest, uri=uri, nonce=nonce,
                                                  nonce_count=nonce_count,
                                                  client_nonce=client_nonce)

    return 'Digest %s' % format_parts(username=username, realm=realm, nonce=nonce, uri=uri,
                                      response=request_digest, algorithm='MD5', opaque=opaque,
                                      qop='auth', nc='%08x' % nonce_count, cnonce=client_nonce)