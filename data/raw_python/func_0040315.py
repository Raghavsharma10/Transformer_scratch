def _jws_signature(signdata, privkey, algorithm):
    """
    Produce a base64-encoded JWS signature based on the signdata
    specified, the privkey instance, and the algorithm passed.
    """
    signature = algorithm.sign(privkey, signdata)
    return base64url_encode(signature)