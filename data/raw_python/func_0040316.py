def sign_serialize(privkey, expire_after=3600, requrl=None,
                   algorithm_name=DEFAULT_ALGO, **kwargs):
    """
    Produce a JWT compact serialization by generating a header, payload,
    and signature using the privkey and algorithm specified.

    The privkey object must contain at least a member named pubkey.

    The parameter expire_after is used by the server to reject the payload
    if received after current_time + expire_after. Set it to None to disable
    its use.

    The parameter requrl is optionally used by the server to reject the
    payload if it is not delivered to the proper place, e.g. if requrl
    is set to https://example.com/api/login but sent to a different server
    or path then the receiving server should reject it.

    Any other parameters are passed as is to the payload.
    """
    assert algorithm_name in ALGORITHM_AVAILABLE

    algo = ALGORITHM_AVAILABLE[algorithm_name]
    addy = algo.pubkey_serialize(privkey.pubkey)

    header = _jws_header(addy, algo).decode('utf8')
    payload = _build_payload(expire_after, requrl, **kwargs)
    signdata = "{}.{}".format(header, payload)
    signature = _jws_signature(signdata, privkey, algo).decode('utf8')

    return "{}.{}".format(signdata, signature)