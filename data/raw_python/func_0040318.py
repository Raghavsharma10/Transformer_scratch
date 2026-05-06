def multisig_validate_deserialize(rawmsg, requrl=None, check_expiration=True,
                                  decode_payload=True,
                                  algorithm_name=DEFAULT_ALGO):
    """
    Validate a general JSON serialization and return the headers and
    payload if all the signatures are good.

    If check_expiration is False, the payload will be accepted even if
    expired.

    If decode_payload is True then this function will attempt to decode
    it as JSON, otherwise the raw payload will be returned. Note that
    it is always decoded from base64url.
    """
    assert algorithm_name in ALGORITHM_AVAILABLE

    algo = ALGORITHM_AVAILABLE[algorithm_name]

    data = json.loads(rawmsg)
    payload64 = data.get('payload', None)
    signatures = data.get('signatures', None)
    if payload64 is None or not isinstance(signatures, list):
        raise InvalidMessage('must contain "payload" and "signatures"')
    if not len(signatures):
        raise InvalidMessage('no signatures')

    try:
        payload, sigs = _multisig_decode(payload64, signatures, decode_payload)
    except Exception as err:
        raise InvalidMessage(str(err))

    all_valid = True
    try:
        for entry in sigs:
            valid = _verify_signature(algorithm=algo, **entry)
            all_valid = all_valid and valid
    except Exception as err:
        raise InvalidMessage('failed to verify signature: {}'.format(err))

    if not all_valid:
        return None, None

    if decode_payload:
        _verify_payload(payload, check_expiration, requrl)
    return [entry['header'] for entry in sigs], payload