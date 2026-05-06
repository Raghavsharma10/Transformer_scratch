def tokhex(length=10, urlsafe=False):
    """
    Return a random string in hexadecimal
    """
    if urlsafe is True:
        return secrets.token_urlsafe(length)
    return secrets.token_hex(length)