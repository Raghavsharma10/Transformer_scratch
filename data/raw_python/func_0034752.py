def calculate_request_digest(method, partial_digest, digest_response=None,
                             uri=None, nonce=None, nonce_count=None, client_nonce=None):
    '''
    Calculates a value for the 'response' value of the client authentication request.
    Requires the 'partial_digest' calculated from the realm, username, and password.

    Either call it with a digest_response to use the values from an authentication request,
    or pass the individual parameters (i.e. to generate an authentication request).
    '''
    if digest_response:
        if uri or nonce or nonce_count or client_nonce:
            raise Exception("Both digest_response and one or more "
                            "individual parameters were sent.")
        uri = digest_response.uri
        nonce = digest_response.nonce
        nonce_count = digest_response.nc
        client_nonce=digest_response.cnonce
    elif not (uri and nonce and (nonce_count != None) and client_nonce):
        raise Exception("Neither digest_response nor all individual parameters were sent.")

    ha2 = md5.md5("%s:%s" % (method, uri)).hexdigest()
    data = "%s:%s:%s:%s:%s" % (nonce, "%08x" % nonce_count, client_nonce, 'auth', ha2)
    kd = md5.md5("%s:%s" % (partial_digest, data)).hexdigest()
    return kd