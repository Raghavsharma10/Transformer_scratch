def decode_jwt(encoded_token):
    """
    Returns the decoded token from an encoded one. This does all the checks
    to insure that the decoded token is valid before returning it.
    """
    secret = config.decode_key
    algorithm = config.algorithm
    audience = config.audience
    return jwt.decode(encoded_token, secret, algorithms=[algorithm], audience=audience)