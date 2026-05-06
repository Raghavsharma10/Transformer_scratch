def to_char(token):
    """Transforms the ASCII control character symbols to their real char.

    Note: If the token is not an ASCII control character symbol, just
    return the token.

    Keyword arguments:
    token -- the token to transform

    """
    if ord(token) in _range(9216, 9229 + 1):
        token = _unichr(ord(token) - 9216)

    return token