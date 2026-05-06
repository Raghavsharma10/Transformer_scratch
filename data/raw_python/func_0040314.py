def _jws_payload(expire_at, requrl=None, **kwargs):
    """
    Produce a base64-encoded JWS payload.

    expire_at, if specified, must be a number that indicates
    a timestamp after which the message must be rejected.

    requrl, if specified, is used as the "audience" according
    to the JWT spec.

    Any other parameters are passed as is to the payload.
    """
    data = {
        'exp': expire_at,
        'aud': requrl
    }
    data.update(kwargs)

    datajson = json.dumps(data, sort_keys=True).encode('utf8')
    return base64url_encode(datajson)