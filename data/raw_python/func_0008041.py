def get_body_hash(params):
    """
    Returns BASE64 ( HASH (text) ) as described in OAuth2 MAC spec.

    http://tools.ietf.org/html/draft-ietf-oauth-v2-http-mac-00#section-3.2
    """
    norm_params = get_normalized_params(params)

    return binascii.b2a_base64(hashlib.sha1(norm_params).digest())[:-1]