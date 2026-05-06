def decode_header(auth_header, client_id, client_secret):
    """
    A function that threads the header through decoding and returns a tuple
    of the token and payload if successful. This does not fully authenticate
    a request.
    :param auth_header:
    :param client_id:
    :param client_secret:
    :return: (token, profile)
    """
    return _decode_header(
        _well_formed(
            _has_token(_has_bearer(_has_header(auth_header)))),
        client_id, client_secret)