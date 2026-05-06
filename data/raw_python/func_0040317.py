def multisig_sign_serialize(privkeys, expire_after=3600, requrl=None,
                            algorithm_name=DEFAULT_ALGO, **kwargs):
    """
    Produce a general JSON serialization by generating a header, payload,
    and multiple signatures using the list of private keys specified.
    All the signatures will be performed using the same algorithm.

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

    payload = _build_payload(expire_after, requrl, **kwargs)
    result = {"payload": payload, "signatures": []}
    algo = ALGORITHM_AVAILABLE[algorithm_name]

    for pk in privkeys:
        addy = algo.pubkey_serialize(pk.pubkey)
        header = _jws_header(addy, algo).decode('utf8')
        signdata = "{}.{}".format(header, payload)
        signature = _jws_signature(signdata, pk, algo).decode('utf8')
        result["signatures"].append({
            "protected": header,
            "signature": signature})

    return json.dumps(result)