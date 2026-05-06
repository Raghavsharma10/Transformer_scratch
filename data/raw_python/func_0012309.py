def read_session_token(secret_key, token):
    """
        This function verifies the token using the secret key and returns its
        contents.
    """
    return jwt.decode(token.encode('utf-8'), secret_key,
        algorithms=[token_encryption_algorithm()]
    )