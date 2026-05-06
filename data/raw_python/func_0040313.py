def _jws_header(keyid, algorithm):
    """Produce a base64-encoded JWS header."""
    data = {
        'typ': 'JWT',
        'alg': algorithm.name,
        # 'kid' is used to indicate the public part of the key
        # used during signing.
        'kid': keyid
    }

    datajson = json.dumps(data, sort_keys=True).encode('utf8')
    return base64url_encode(datajson)