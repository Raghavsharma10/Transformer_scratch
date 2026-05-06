def validate_deserialize(rawmsg, requrl=None, check_expiration=True,
                         decode_payload=True, algorithm_name=DEFAULT_ALGO):
    """
    Validate a JWT compact serialization and return the header and
    payload if the signature is good.

    If check_expiration is False, the payload will be accepted even if
    expired.

    If decode_payload is True then this function will attempt to decode
    it as JSON, otherwise the raw payload will be returned. Note that
    it is always decoded from base64url.
    """
    assert algorithm_name in ALGORITHM_AVAILABLE
    algo = ALGORITHM_AVAILABLE[algorithm_name]

    segments = rawmsg.split('.')
    if len(segments) != 3 or not all(segments):
        raise InvalidMessage('must contain 3 non-empty segments')

    header64, payload64, cryptoseg64 = segments
    try:
        signature = base64url_decode(cryptoseg64.encode('utf8'))
        payload_data = base64url_decode(payload64.encode('utf8'))
        header_data = base64url_decode(header64.encode('utf8'))
        header = json.loads(header_data.decode('utf8'))
        if decode_payload:
            payload = json.loads(payload_data.decode('utf8'))
        else:
            payload = payload_data
    except Exception as err:
        raise InvalidMessage(str(err))

    try:
        valid = _verify_signature(
            '{}.{}'.format(header64, payload64),
            header,
            signature,
            algo)
    except Exception as err:
        raise InvalidMessage('failed to verify signature: {}'.format(err))

    if not valid:
        return None, None

    if decode_payload:
        _verify_payload(payload, check_expiration, requrl)
    return header, payload